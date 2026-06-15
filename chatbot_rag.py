"""
chatbot_rag.py
--------------
RAG chatbot powered by:
  • Groq (llama-3.3-70b-versatile) — free, fast LLM
  • Pinecone                        — vector store
  • HuggingFace Embeddings          — all-MiniLM-L6-v2, free & local

Assistant persona: "Lumi" — a friendly, professional AI assistant.

Run:
    streamlit run chatbot_rag.py --server.fileWatcherType none
"""

import os
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lumi Assistant",
    page_icon="✨",
    layout="centered",
)

# ── Custom CSS — clean, light, professional ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&family=Inter:wght@400;500&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
      background: #ffffff;
      color: #1f2937;
      font-family: 'Inter', sans-serif;
  }
  [data-testid="stHeader"] { background: transparent; }

  h1 { font-family: 'Poppins', sans-serif !important; font-weight: 600 !important;
       color: #4f46e5 !important; letter-spacing: 0.01em; }

  .subtitle { font-family: 'Inter', sans-serif; color: #6b7280;
              font-size: 0.95rem; margin-top: -12px; margin-bottom: 28px; }

  [data-testid="stChatMessage"] {
      background: #f9fafb !important;
      border-radius: 14px !important;
      border: 1px solid #e5e7eb !important;
      margin-bottom: 10px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
      background: #eef2ff !important;
      border-left: 3px solid #4f46e5 !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
      background: #f0fdf4 !important;
      border-left: 3px solid #10b981 !important;
  }
  [data-testid="stChatInput"] textarea {
      background: #ffffff !important;
      color: #1f2937 !important;
      border: 1px solid #d1d5db !important;
      border-radius: 10px !important;
  }
  [data-testid="stSidebar"] {
      background: #f9fafb !important;
      border-right: 1px solid #e5e7eb;
  }
  [data-testid="stSidebar"] * { color: #374151 !important; }

  /* Slider accent */
  [data-testid="stSlider"] [role="slider"] { background-color: #4f46e5 !important; }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("✨ Lumi")
st.markdown('<p class="subtitle">Your AI assistant for document Q&A.</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Free at https://console.groq.com — no billing required",
    )

    st.markdown("---")
    st.markdown("### 🤖 About Lumi")
    st.markdown("Lumi is a helpful, clear, and friendly AI assistant that answers questions based on your uploaded documents.")

    st.markdown("---")
    st.markdown("### 📂 Documents")
    st.markdown("Add PDFs or `.txt` files to `documents/`, then run:")
    st.code("python ingestion.py", language="bash")

    st.markdown("---")
    top_k = st.slider("Chunks to retrieve (k)", 1, 8, 3)
    score_threshold = st.slider("Similarity threshold", 0.1, 0.9, 0.4, 0.05)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Validate Groq key ─────────────────────────────────────────────────────────
if not groq_key:
    st.warning(
        "👈 Enter your **Groq API key** in the sidebar to begin.\n\n"
        "Get one free (no credit card) at [console.groq.com](https://console.groq.com) → API Keys → Create."
    )
    st.stop()

os.environ["GROQ_API_KEY"] = groq_key

# ── Init Pinecone + embeddings (cached) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vector_store():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX_NAME", "langchain-sample-index")
    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PineconeVectorStore(index=index, embedding=embeddings)

@st.cache_resource(show_spinner="Starting Lumi...")
def load_llm(api_key: str):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.5,
        max_tokens=512,
    )

try:
    vector_store = load_vector_store()
    llm = load_llm(groq_key)
except Exception as e:
    st.error(f"Failed to initialise: {e}")
    st.stop()

# ── Lumi system prompt ─────────────────────────────────────────────────────────
LUMI_SYSTEM = """You are Lumi, a helpful and friendly AI assistant.

Your tone is:
• Clear, professional, and approachable.
• Concise and well-organized — use short paragraphs or bullet points when helpful.
• Friendly but not overly casual; confident and accurate.
• You speak naturally, like a helpful colleague who knows the material well.

RULES:
1. Answer ONLY using the retrieved context below. If the answer isn't in the context, say so politely and suggest rephrasing the question.
2. Keep answers concise and to the point unless the question requires detail.
3. Do not make up information that isn't in the context.
4. You may refer to yourself as "Lumi" when natural.

Retrieved context:
{context}"""

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask Lumi anything about your documents...")

if prompt:
    st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant chunks
    with st.spinner("Searching your documents..."):
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )
        try:
            docs = retriever.invoke(prompt)
        except Exception:
            docs = []

    context_text = "\n\n---\n\n".join(d.page_content for d in docs) if docs else ""
    if not context_text:
        context_text = "No relevant documents were found for this query."

    system_msg = SystemMessage(LUMI_SYSTEM.format(context=context_text))
    recent = st.session_state.messages[-6:]
    messages_to_send = [system_msg] + recent

    with st.chat_message("assistant"):
        with st.spinner("Lumi is thinking..."):
            try:
                response = llm.invoke(messages_to_send)
                answer = response.content
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"

        st.markdown(answer)

        if docs:
            with st.expander(f"📄 {len(docs)} source chunk(s) used", expanded=False):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "Unknown")
                    page   = doc.metadata.get("page", "")
                    label  = f"**Source {i}** — `{os.path.basename(source)}`"
                    if page != "":
                        label += f", page {int(page)+1}"
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    if i < len(docs):
                        st.divider()

    st.session_state.messages.append(AIMessage(answer))