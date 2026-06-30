import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()
key=os.getenv("OPENAI_API_KEY")

def embeddings():
    model=OpenAIEmbeddings(model="text-embedding-3-small")
    return model