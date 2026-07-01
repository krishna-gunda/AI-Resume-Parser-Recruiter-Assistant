from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def chat_model():
    model = ChatOpenAI(
        model="gpt-4.1-mini",     
        temperature=0
    )

    return model