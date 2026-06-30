from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(document)

    return chunks