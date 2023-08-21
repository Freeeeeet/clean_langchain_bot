import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from pyrogram import Client, filters


load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(temperature=0)

db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./vectorstore")

query = "Состав продукта Prostatricum"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# print(qa.run(query))

bot = Client(
    "some_bot_name",
    api_id=os.getenv("API_ID"),
    api_hash=os.getenv("API_HASH"),
    bot_token=os.getenv("TOKEN")
)


@bot.on_message(filters.private & filters.text)
def handle_input(client, message):
    try:
        msgs = message.text
        K = message.reply_text("Минуточку, сейчас уточню...")

        query = msgs


        docs = db.similarity_search(query)
        first = docs[0].page_content
        print(first)
        message.reply_text(first)

        second = qa.run(query)
        print(second)
        message.reply_text(second)

        K.delete()

    except Exception as e:

        notworking = "https://st4.depositphotos.com/5365202/37818/v/450/depositphotos_378186364-stock-illustration-hand-drawn-vector-cartoon-illustration.jpg"
        message.reply_photo(
            photo=notworking,
            caption=f"Сейчас не могу ответить, ведутся технические работы, попробуйте чуть позже или обратитесь в нашу тех. поддержку")

bot.run()