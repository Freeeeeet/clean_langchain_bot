import os
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI

from langchain.embeddings import HuggingFaceInstructEmbeddings

load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl",
    )


raw_documents = TextLoader('./VARILUX_RUSSIA_GEB_rev_3.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, embeddings, persist_directory="./vectorstore")
db.persist()
db = None
