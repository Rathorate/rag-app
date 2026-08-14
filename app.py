import os
import tempfile

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Setup ----------
load_dotenv()


def get_secret(name: str):
    # Locally this comes from .env; on Streamlit Cloud it comes from
    # the Secrets manager, which may not always populate os.environ.
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


GATEWAY_URL = get_secret("GATEWAY_URL")
GATEWAY_API_KEY = get_secret("GATEWAY_API_KEY")

st.set_page_config(page_title="Talk to my documents", page_icon="📄")
st.title("📄 Talk to my documents")
st.caption("Upload a PDF, then ask questions grounded in its content.")

if not GATEWAY_URL or not GATEWAY_API_KEY:
    st.error("Missing GATEWAY_URL or GATEWAY_API_KEY. Check your .env file.")
    st.stop()


def ask_gateway(prompt: str) -> str:
    """Send a prompt to the n8n gateway and return the generated text."""
    response = requests.post(
        GATEWAY_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": GATEWAY_API_KEY,
        },
        json={"message": prompt, "mode": "auto"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("output"):
        raise RuntimeError(f"Gateway returned no output: {data}")
    return data["output"]


# ---------- Session state ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "doc_matrix" not in st.session_state:
    st.session_state.doc_matrix = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Step 1: Upload + process document ----------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and st.session_state.chunks is None:
    with st.spinner("Reading and indexing your document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(pages)

        # Build a TF-IDF search index over the chunk texts (no neural
        # embeddings, no torch — just keyword/term-frequency based search).
        texts = [c.page_content for c in chunks]
        vectorizer = TfidfVectorizer(stop_words="english")
        doc_matrix = vectorizer.fit_transform(texts)

        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.doc_matrix = doc_matrix
        os.remove(tmp_path)

    st.success(f"Indexed {len(chunks)} chunks from your document. Ask away!")


def retrieve_relevant_chunks(question: str, k: int = 4):
    query_vec = st.session_state.vectorizer.transform([question])
    scores = cosine_similarity(query_vec, st.session_state.doc_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return [st.session_state.chunks[i] for i in top_indices]


# ---------- Step 2: Chat interface ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your document...")

if question:
    if st.session_state.chunks is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_docs = retrieve_relevant_chunks(question)

                context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

                prompt = (
                    "Answer the question using ONLY the context below. "
                    "If the answer isn't in the context, say you don't know.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )

                try:
                    answer = ask_gateway(prompt)
                except Exception as e:
                    answer = f"Something went wrong calling the gateway: {e}"

                st.markdown(answer)

                with st.expander("Show source excerpts used"):
                    for i, doc in enumerate(relevant_docs, start=1):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Excerpt {i} (page {page}):**")
                        st.text(doc.page_content[:400] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})