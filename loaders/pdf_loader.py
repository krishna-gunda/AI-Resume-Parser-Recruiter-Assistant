import langchain
import langchain_community
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader

def loader(path):
    loader=PyPDFLoader(path)
    document=loader.load()
    return document


