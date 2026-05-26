# 🔍 Semantic Similarity Search

Simple semantic search using HuggingFace embeddings + FAISS, built with Streamlit.
Works on Windows, Mac, and Linux with no C++ build tools needed.

## Tech Stack
| Component   | Library |
|-------------|---------|
| Embeddings  | `sentence-transformers` → `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB   | `faiss-cpu` (Facebook AI Similarity Search) |
| UI          | `streamlit` |

---

## 🚀 Run Locally

```bash
# 1. Unzip and enter the folder
cd semantic-search

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Opens at http://localhost:8501

---

## ☁️ Deploy Free → Get a Public URL

### Streamlit Community Cloud (recommended)
1. Push this folder to a public GitHub repo
2. Go to https://share.streamlit.io → New App
3. Select repo, branch, and set `app.py` as the main file
4. Click Deploy — URL ready in ~2 min

### Hugging Face Spaces
1. Create a Space at https://huggingface.co/spaces (SDK = Streamlit)
2. Upload `app.py`, `requirements.txt`, `README.md`
3. Auto-builds and gives a public URL
