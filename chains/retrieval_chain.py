from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def retrieval_chain(retriever, prompt, llm):

    chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain