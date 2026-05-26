import streamlit as st
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Similarity Search")
st.caption("Natural language search powered by HuggingFace embeddings + FAISS vector index")

# ── Sample documents ──────────────────────────────────────────────────────────
SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "The solar system consists of the Sun and eight planets, including Earth and Mars.",
    "Deep learning uses neural networks with many layers to model complex patterns in data.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
    "The stock market allows investors to buy and sell shares of publicly traded companies.",
    "DNA carries the genetic information that determines the traits of living organisms.",
    "Quantum computing uses quantum phenomena to process information exponentially faster.",
    "The Renaissance was a cultural movement in Europe from the 14th to 17th century.",
    "Photosynthesis is the process by which plants convert sunlight into glucose and oxygen.",
    "Blockchain is a distributed ledger technology that enables secure, transparent transactions.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Natural language processing allows computers to understand and generate human language.",
    "Electric vehicles use battery-powered motors instead of internal combustion engines.",
    "The Internet of Things connects everyday devices to the internet for data exchange.",
    "Yoga combines physical postures, breathing exercises, and meditation for overall wellbeing.",
    "Supply chain management coordinates the flow of goods from raw materials to consumers.",
    "Vaccines train the immune system to recognize and fight specific pathogens.",
    "Jazz music originated in African-American communities in New Orleans in the early 20th century.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
]

# ── Session state: document list ──────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = list(SAMPLE_DOCS)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HuggingFace model (first run only)…")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ── Build FAISS index from current documents ──────────────────────────────────
@st.cache_data(show_spinner="Building vector index…")
def build_index(docs: tuple):
    embeddings = model.encode(list(docs), convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)                      # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])      # inner product = cosine after normalise
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_index(tuple(st.session_state.documents))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Add Your Own Document")
    new_doc = st.text_area("Paste any text:", height=130,
                           placeholder="Type or paste a sentence / paragraph…")
    if st.button("➕ Add Document", use_container_width=True):
        text = new_doc.strip()
        if text:
            st.session_state.documents.append(text)
            build_index.clear()          # invalidate cache so index rebuilds
            st.success(f"Added! Total docs: {len(st.session_state.documents)}")
            st.rerun()
        else:
            st.warning("Please enter some text.")

    st.divider()
    st.metric("Documents in index", len(st.session_state.documents))
    top_k = st.slider("Results to show", 1, 10, 5)

    st.divider()
    if st.button("🔄 Reset to sample docs", use_container_width=True):
        st.session_state.documents = list(SAMPLE_DOCS)
        build_index.clear()
        st.rerun()

# ── Search bar ────────────────────────────────────────────────────────────────
query = st.text_input("🔎 Enter your search query:",
                      placeholder="e.g. How do plants make food?")

if st.button("Search", type="primary") or query:
    if query.strip():
        q_emb = model.encode([query.strip()], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        k = min(top_k, len(st.session_state.documents))
        scores, indices = index.search(q_emb, k)

        st.subheader(f"Top {k} Results")
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            similarity = float(score)          # already 0-1 after L2 normalisation
            icon = "🟢" if similarity > 0.55 else "🟡" if similarity > 0.35 else "🔴"
            doc_text = st.session_state.documents[idx]

            with st.container(border=True):
                c1, c2 = st.columns([7, 1])
                with c1:
                    st.markdown(f"**#{rank}** {doc_text}")
                with c2:
                    st.markdown(f"{icon} **{similarity:.0%}**")
                st.progress(max(0.0, min(similarity, 1.0)),
                            text=f"Cosine similarity: {similarity:.4f}")
    else:
        st.info("Type a query and press Search.")

# ── Browse all docs ───────────────────────────────────────────────────────────
with st.expander("📚 Browse all documents in the index"):
    for i, doc in enumerate(st.session_state.documents, 1):
        st.markdown(f"**{i}.** {doc}")
