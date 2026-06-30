import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import loaders
from loaders.pdf_loader import loader
import splitter
from splitter.text_splitter import splitter
import embeddings
from embeddings.openai_embeddings import embeddings
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings



load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

def main():
    document=loader(r"uploaded files\G_Krishna_AIML_Engineer.pdf")
    chunks=splitter(document)
    model=embeddings()
    print("Embedding model loaded successfully!")
    

if __name__ == "__main__":
    main()   