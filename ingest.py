import os
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI

load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

raw_documents = TextLoader('./RA_new_inst_Proststricum_RU_GEB_rev_3.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory="./vectorstore")
