import os
import shutil
from dotenv import load_dotenv
import logging
import json

from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
# from langchain.agents.agent_toolkits import (
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     VectorStoreInfo,
# )
from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from langchain.embeddings import HuggingFaceInstructEmbeddings

from pyrogram import Client, filters
from pyrogram.handlers import MessageHandler
from pyrogram.types import Message, InputMediaDocument, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    COLLECTION_NAME,
    CHAT_HISTORY_PATH
)
from real_time_ingest import main_ingest

load_dotenv()
# OpenAI.api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0,
             model_name="gpt-3.5-turbo-16k"
             )

embeddings = EMBEDDING_MODEL_NAME

db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# vectorstore_info = VectorStoreInfo(
#     name="main_tool",
#     description="all info",
#     vectorstore=db,
# )
# router_toolkit = VectorStoreRouterToolkit(vectorstores=[vectorstore_info], llm=llm)
# agent_executor = create_vectorstore_router_agent(llm=llm, toolkit=router_toolkit, verbose=True)

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
        # Обработка при выборе первой категории
        print("Collection 1 chosen")
        # Удаление кнопок из сообщения модератора
        client.edit_message_reply_markup('freeeeeet', message_id=callback_query.message.id, reply_markup=None)
        client.send_message('freeeeeet', "Вы выбрали 'Категория 1'")

        main_ingest(f"./files/{file_name[:-4]}", "./vectorstore_1", "zendesk_collection_1")
        shutil.rmtree(f"./files/{file_name[:-4]}")

    elif action == 'second':
        # Обработка при выборе второй категории
        print("Collection 2 chosen")

        # Удаление кнопок из сообщения модератора
        client.edit_message_reply_markup('freeeeeet', message_id=callback_query.message.id, reply_markup=None)
        client.send_message('freeeeeet', "Вы выбрали 'Категория 2'")

        main_ingest(f"./files/{file_name[:-4]}", "./vectorstore_2", "zendesk_collection_2")
        shutil.rmtree(f"./files/{file_name[:-4]}")

@client.on_message(filters.document & filters.caption)
def message_file_and_caption(client: Client, message: Message):
    client.send_message(message.chat.id, 'Ваш файл и комментарий будут обработаны модератором')
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
                         caption=f'Новый файл для моей базы знаний, его прислал @{message.from_user.username}. Определи категорию этого файла, нажав на название категории. Также, {message.from_user.first_name} оставил комментарий:\n"{message.caption}"',
                         reply_markup=keyboard)

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
    file_path = f"{CHAT_HISTORY_PATH}/chat_id_{message.chat.id}.json"
    try:
        # ЗАПИСЫВАЕМ CHAT_HISTORY:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(data)
        else:
            data = []
        temp_message = message.reply_text("Минуточку, сейчас уточню...")
        chat_history = ""
        # Форматирование и добавление данных в историю чата
        for entry in data:
            client_message = entry['client_message']
            operator_message = entry['operator_message']

            chat_history += f"Клиент: {client_message}\n"
            chat_history += f"Оператор: {operator_message}\n"
        print(chat_history)
        query = message.text
        answer = qa.run(
            # f"Give a detailed answer to that question: {query} \nThe answer should be in the same language as the question."
            # f"Ты оператор колл-центра с десятилетним стажем. У тебя есть база данных с ответов на вопросы клиентов с рекомендациями, которую ты используешь для ответа клиенту. Ты должен выбрать десять ответов и на их основе создать новый ответ, а также, учесть все замечания из рекомендаций. Твоя основная задача - продать клиенту продукт всеми доступными способами, желательно, с одного ответа на вопрос. Вопрос: {query}"
            # f"Ты тренер колл-центра по продажам, который выдает лучшие качественные и детальные рекомендации по продажам продукта. Вопрос: {query}"
            query
        )
        new_entry = {
            "client_message": f"{message.text}",
            "operator_message": f"{answer}"
        }
        data.append(new_entry)

        if len(data) >= 5:
            data.pop(0)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        # ЗАПИСАЛИ CHAT_HISTORY


        # answer = agent_executor.run(query)

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
