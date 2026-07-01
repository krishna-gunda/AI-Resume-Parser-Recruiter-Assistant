from langchain_core.prompts import ChatPromptTemplate

def resume_prompt():
    prompt = ChatPromptTemplate.from_template("""
You are an AI Resume Parser.

Answer ONLY using the resume context.

If the answer is not available in the context, say:
"I couldn't find that information in the resume."

Context:
{context}

Question:
{input}

Answer:
""")

    return prompt