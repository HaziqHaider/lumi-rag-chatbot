# check_pinecone.py
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
indexes = pc.list_indexes()
for i in indexes:
    print(i["name"])