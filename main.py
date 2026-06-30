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
     

    

if __name__ == "__main__":
    main()   