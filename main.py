import os
from dotenv import load_dotenv
import logging

from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from langchain.embeddings import HuggingFaceInstructEmbeddings

from pyrogram import Client, filters
from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)

load_dotenv()
# OpenAI.api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

##################################################
#embeddings = EMBEDDING_MODEL_NAME

embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )

# model_id = "lmsys/vicuna-13b-v1.3"
# model_id = "daryl149/llama-2-7b-chat-hf"
# model_id = "Photolens/llama-2-7b-langchain-chat"


# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
#
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
#
# generation_config = GenerationConfig.from_pretrained(model_id)

# pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_length=8192,
#         temperature=0,
#         top_p=0.95,
#         repetition_penalty=1.15,
#         generation_config=generation_config,
#     )

# llm = HuggingFacePipeline(pipeline=pipe)

#################################################
db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

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
        # message.reply_text(first)

        second = qa.run(f"Give a detailed answer to that question: {query} \nThe answer should be in the same language as the question.")
        print(second)
        message.reply_text(second)

        K.delete()

    except Exception as e:
        logging.exception("An error occurred during message processing: %s", e)
        notworking = "https://st4.depositphotos.com/5365202/37818/v/450/depositphotos_378186364-stock-illustration-hand-drawn-vector-cartoon-illustration.jpg"
        message.reply_photo(
            photo=notworking,
            caption=f"Сейчас не могу ответить, ведутся технические работы, попробуйте чуть позже или обратитесь в нашу тех. поддержку")

bot.run()
