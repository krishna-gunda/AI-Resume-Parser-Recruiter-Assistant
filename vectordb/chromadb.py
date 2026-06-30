import os
import shutil
from embeddings.openai_embeddings import embedding_model
from langchain_chroma import Chroma

def creating_embeddings(chunks):

    # Path where ChromaDB is stored
    db_path = "./vectordb/chroma_db"

    # Delete the existing database if it already exists
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print("Previous vector database deleted successfully.")

    # Load the embedding model
    model = embedding_model()

    # Create a new vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=model,
        persist_directory=db_path
    )

    print("New vector database created successfully.")

    return vector_db