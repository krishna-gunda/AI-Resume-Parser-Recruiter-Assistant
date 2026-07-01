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
import chains
from chains.retrieval_chain import retrieval_chain

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

    llm = chat_model()

    prompt = resume_prompt()

    rag_chain = retrieval_chain(
    retriever=retriever_obj,
    prompt=prompt,
    llm=llm)
    

    question = "What is the college name that krishna studied"

    response = rag_chain.invoke(question)

    print(response)
    

if __name__ == "__main__":
    main()   