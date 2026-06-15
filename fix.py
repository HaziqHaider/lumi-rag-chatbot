import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(override=True)

api_key = os.environ.get("PINECONE_API_KEY", "")
index_name = os.environ.get("PINECONE_INDEX_NAME", "")

print("API KEY  :", "FOUND" if api_key else "MISSING")
print("INDEX    :", index_name if index_name else "MISSING")

if not api_key:
    print("ERROR: No API key. Check your .env file.")
    exit(1)

try:
    pc = Pinecone(api_key=api_key)
    names = [i["name"] for i in pc.list_indexes()]
    print("INDEXES  :", names if names else "(empty — no indexes exist)")
except Exception as e:
    print("PINECONE ERROR:", e)