import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import loaders
from loaders.pdf_loader import loader
import splitter
from splitter.text_splitter import splitter
import embeddings
from embeddings.openai_embeddings import embedding_model
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import vectordb
from vectordb.chromadb import creating_embeddings
import retriever
from retriever.chroma_retriever import retriever
import query
from query.query import retrieve_documents



load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

def main():
    document=loader(r"uploaded files\G_Krishna_AIML_Engineer.pdf")
    chunks=splitter(document)
    print(f'length of chunks {len(chunks)}')
    model=embedding_model()
    print("Embedding model loaded successfully!")

    vector_db=creating_embeddings(chunks)
    collection = vector_db._collection.get(include=["embeddings"]) # here in the chroma db it store the 
    #data in the for of ids and there embeddings so we include only embeddings

    print("Number of vectors:", len(collection["embeddings"]))
    print("Dimensions of first vector:", len(collection["embeddings"][0]))


    print("===================================================================================")

    related_ans = retriever(vector_db)

    query = "What is the highest qualification of Krishna?"

    docs = retrieve_documents(related_ans, query)

    for i, chunk in enumerate(docs, start=1):
        print(f"\nChunk {i}")
        print(chunk.page_content)


    

if __name__ == "__main__":
    main()   