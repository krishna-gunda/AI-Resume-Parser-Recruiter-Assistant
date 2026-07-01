import os
from embeddings.openai_embeddings import embedding_model
from langchain_chroma import Chroma

def creating_embeddings(chunks):

    # Path where ChromaDB is stored
    db_path = "./vectordb/chroma_db"

    # Load the embedding model
    model = embedding_model()

    # If the vector database already exists, load it
    if os.path.exists(db_path):
        print("Existing vector database found. Loading embeddings...")

        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=model
        )

        return vector_db

    # Otherwise, create a new vector database
    print("No existing vector database found. Creating embeddings...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=model,
        persist_directory=db_path
    )

    print("New vector database created successfully.")

    return vector_db