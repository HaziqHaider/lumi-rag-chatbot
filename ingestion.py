"""
ingestion.py
------------
Loads PDFs and .txt files from the `documents/` folder,
splits them into chunks, embeds with HuggingFace (free, local),
and upserts into a Pinecone serverless index.

Run once (or whenever you add new documents):
    python ingestion.py
"""

import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Pinecone setup ────────────────────────────────────────────────────────────
api_key = os.environ.get("PINECONE_API_KEY", "")
if not api_key:
    print("❌  PINECONE_API_KEY not found in .env — check your .env file.")
    exit(1)

pc = Pinecone(api_key=api_key)
index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-chatbot-index")

print(f"🔍  Pinecone API key loaded: {api_key[:12]}...")
print(f"🗂️   Target index name     : '{index_name}'")

# List existing indexes
existing_indexes = pc.list_indexes()
existing_names = [i["name"] for i in existing_indexes]
print(f"📋  Existing indexes       : {existing_names if existing_names else '(none)'}")

if index_name in existing_names:
    print(f"✅  Index '{index_name}' already exists — deleting and recreating fresh.")
    pc.delete_index(index_name)
    time.sleep(3)

print(f"🔨  Creating index '{index_name}' ...")
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

print("⏳  Waiting for index to be ready...")
for _ in range(30):
    status = pc.describe_index(index_name).status
    print(f"    Status: {status}")
    if status.get("ready"):
        break
    time.sleep(2)

print(f"✅  Index '{index_name}' is ready.")

index = pc.Index(index_name)

# ── Embeddings (free, runs locally) ──────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ── Load documents from `documents/` folder ───────────────────────────────────
docs_dir = "documents"
os.makedirs(docs_dir, exist_ok=True)

all_docs = []

# PDFs
pdf_loader = PyPDFDirectoryLoader(docs_dir)
pdf_docs = pdf_loader.load()
all_docs.extend(pdf_docs)
print(f"Loaded {len(pdf_docs)} PDF pages.")

# .txt files
txt_loader = DirectoryLoader(
    docs_dir,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    silent_errors=True,
)
txt_docs = txt_loader.load()
all_docs.extend(txt_docs)
print(f"Loaded {len(txt_docs)} text file(s).")

if not all_docs:
    print("⚠  No documents found in `documents/`. Add PDFs or .txt files and re-run.")
    exit(0)

# ── Split into chunks ─────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)
chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks.")

# ── Upsert into Pinecone ──────────────────────────────────────────────────────
uuids = [f"doc-{i}" for i in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)
print(f"✅  Successfully ingested {len(chunks)} chunks into Pinecone index '{index_name}'.")