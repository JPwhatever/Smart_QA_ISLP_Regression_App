
import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from google import genai

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISLP Regression QA",
    page_icon="📚",
    layout="centered"
)

st.title("📚 ISLP Linear Regression QA System")
st.markdown("Ask any question about linear regression from the ISLP textbook.")

# ── Load everything ────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    with open('regression_chunks.json', 'r') as f:
        chunks = json.load(f)
    embeddings = np.load('regression_embeddings.npy')
    bge_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    tokenized = [c['text'].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    return chunks, embeddings, bge_model, bm25_index

final_chunks, chunk_embeddings, model, bm25 = load_resources()

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ── Retrieval ──────────────────────────────────────────────────────────────
def retrieve(query, top_k=3):
    query_vector = model.encode([BGE_PREFIX + query])
    bge_scores = cosine_similarity(query_vector, chunk_embeddings)[0]
    bge_ranks = bge_scores.argsort()[::-1]

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranks = bm25_scores.argsort()[::-1]

    k = 60
    rrf_scores = {}
    for rank, idx in enumerate(bge_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(k + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(k + rank + 1)

    final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in final_ranking[:top_k]]
    return [final_chunks[i] for i in top_indices]

# ── Generation ─────────────────────────────────────────────────────────────
def ask(question, generator="Gemini"):
    top_chunks = retrieve(question, top_k=3)
    context = "\n\n---\n\n".join([
        f"Section: {c['heading']}\n{c['text']}"
        for c in top_chunks
    ])
    prompt = f"""You are a helpful assistant answering questions about linear regression from the ISLP textbook.
Use ONLY the context provided below. Do not use outside knowledge.
Answer in 3-4 sentences. Do not ask follow up questions.
If the answer is not in the context, say: "This is not covered in the provided sections."

Context:
{context}

Question: {question}

Answer:"""

    try:
        if generator == "Gemini":
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer = response.text
        else:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Generator error — please try Llama 3.1 8B instead. (Error: {str(e)[:100]})"

    return answer, top_chunks
    
# ── UI ─────────────────────────────────────────────────────────────────────
generator = st.radio("Choose generator:", ["Gemini", "Llama 3.1 8B (Groq)"], horizontal=True)

question = st.text_input("Ask a question:", placeholder="e.g. What causes multicollinearity?")

if st.button("Ask") and question:
    with st.spinner("Retrieving and generating answer..."):
        answer, chunks = ask(question, generator)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Sections")
    for c in chunks:
        with st.expander(c["heading"]):
            st.write(c["text"][:500] + "...")

st.markdown("---")
st.markdown("Built with BGE + BM25 hybrid retrieval and Gemini/Llama generation.")
