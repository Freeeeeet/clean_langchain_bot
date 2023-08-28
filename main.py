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
from pyrogram.handlers import MessageHandler
from pyrogram.types import Message, InputMediaDocument, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    COLLECTION_NAME
)
from real_time_ingest import main_ingest

load_dotenv()
# OpenAI.api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

embeddings = EMBEDDING_MODEL_NAME

db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# print(qa.run(query))


# @bot.on_message(filters.private & filters.text)

# @client.on_message()
# def all_message(client: Client, message: Message):
#     message.reply(message.text, reply_to_message_id=message.id)


# client.add_handler(MessageHandler(all_message))

client = Client(
    "zendesk_client",
    api_id=os.getenv("API_ID"),
    api_hash=os.getenv("API_HASH"),
    bot_token=os.getenv("TOKEN")
)


@client.on_callback_query()
def handle_callback_query(client: Client, callback_query: CallbackQuery):
    data = callback_query.data
    action, file_name = data.split(':')
    file_path = f"./files/{file_name[:-4]}"

    if action == 'first':
        # Обработка действия принятия файла
        print("Collection 1 chosen")
        main_ingest(f"./files/{file_name[:-4]}", "./vectorstore", "zendesk_collection")



        # Удаление кнопок из сообщения модератора
        client.edit_message_reply_markup('freeeeeet', message_id=callback_query.message.id, reply_markup=None)

    elif action == 'second':
        # Обработка действия отклонения файла
        print("Collection 2 chosen")

        # Удаление кнопок из сообщения модератора
        client.edit_message_reply_markup('freeeeeet', message_id=callback_query.message.id, reply_markup=None)


@client.on_message(filters.document)
def message_file(client: Client, message: Message):
    client.send_message(message.chat.id, 'Ваш файл будет обработан модератором')
    # client.send_document('freeeeeet', document=message.document.file_id, caption=f'Новый файл для моей базы знаний, его прислал @{message.from_user.username}. Определи категорию этого файла, пожалуйста(кнопка)')
    file_name = message.document.file_name
    file_path = f"./files/{file_name[:-4]}/{file_name[-30:]}"
    client.download_media(message.document.file_id, file_name=file_path)

    # Добавляем кнопки
    keyboard = (InlineKeyboardMarkup
        (
        [
            [
                InlineKeyboardButton("Категория 1",
                                     callback_data=f"first:{file_name}"),

                InlineKeyboardButton("Категория 2",
                                     callback_data=f"second:{file_name}",
                                     )
            ]
        ]
    )
    )
    client.send_document('freeeeeet', document=message.document.file_id,
                         caption=f'Новый файл для моей базы знаний, его прислал @{message.from_user.username}. Определи категорию этого файла',
                         reply_markup=keyboard)

    # message.reply('Ваш файл будет обработан модератором', quote=True)


@client.on_message(filters.text)
def message_text(client: Client, message):
    try:
        msgs = message.text
        temp_message = message.reply_text("Минуточку, сейчас уточню...")
        query = msgs

        answer = qa.run(
            f"Give a detailed answer to that question: {query} \nThe answer should be in the same language as the question.")
        print(answer)
        message.reply_text(answer)

        temp_message.delete()

    except Exception as e:
        logging.exception("An error occurred during message processing: %s", e)
        notworking = "https://st4.depositphotos.com/5365202/37818/v/450/depositphotos_378186364-stock-illustration-hand-drawn-vector-cartoon-illustration.jpg"
        message.reply_photo(
            photo=notworking,
            caption=f"Сейчас не могу ответить, ведутся технические работы, попробуйте чуть позже или обратитесь в нашу тех. поддержку")


# client.add_handler(MessageHandler(message_file, filters=filters.document))
# client.add_handler(MessageHandler(message_text, filters=filters.text))

# client.send_message('freeeeeet', 'Hello Bodya')

# list_media = []
# media_1 = InputMediaDocument('./RA_new_inst_Proststricum_RU_GEB_rev_3.txt', caption='Hello, file:')
# list_media.append(media_1)
# client.send_document('freeeeeet', './RA_new_inst_Proststricum_RU_GEB_rev_3.txt')

client.run()
