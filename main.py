import warnings
warnings.filterwarnings("ignore")

import loaders
from loaders.pdf_loader import loader
import splitter
from splitter.text_splitter import splitter
import embeddings
from embeddings.openai_embeddings import embedding_model
import os
from dotenv import load_dotenv
import vectordb
from vectordb.chromadb import creating_embeddings
import retriever
from retriever.chroma_retriever import retriever
import query
from query.query import retrieve_documents
import models
from models.chat_openai_model import chat_model
import prompts
from prompts.prompt_template import resume_prompt


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

    retriever_obj = retriever(vector_db)

    question = "What is the highest qualification of Krishna?"

    docs = retrieve_documents(retriever_obj, question)

    for i, chunk in enumerate(docs, start=1):
        print(f"\nChunk {i}")
        print(chunk.page_content)


    openai_model=chat_model()
      
    prompt=resume_prompt()

    

if __name__ == "__main__":
    main()   