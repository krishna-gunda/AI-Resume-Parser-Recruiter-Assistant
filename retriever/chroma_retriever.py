import vectordb
from vectordb.chromadb import creating_embeddings


def retriever(vectordb):
    retriever=vectordb.as_retriever(search_kwargs={"k":3})
    return retriever 