from langchain_core.prompts import ChatPromptTemplate

def resume_prompt():

    prompt = ChatPromptTemplate.from_template("""
You are an AI Resume Parser.

Answer the user's question ONLY using the provided resume context.

If the answer is not available in the resume, respond with:
"I couldn't find that information in the resume."

Context:
{context}

Question:
{question}

Answer:
""")

    return prompt