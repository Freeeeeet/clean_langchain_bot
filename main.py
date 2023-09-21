import os
import shutil
from dotenv import load_dotenv
import logging
import json
import pprint

from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo)
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

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
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
search = GoogleSerperAPIWrapper(k=1)

# GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# OpenAI.api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0,
             model_name="gpt-3.5-turbo-16k"
             )
# гугл поиск инициализация
# tools = [
#     Tool(
#         name="Intermediate Answer",
#         func=search.run,
#         description="useful for when you need to ask with search",
#     )
# ]
#
# self_ask_with_search = initialize_agent(
#     tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
# )


embeddings = EMBEDDING_MODEL_NAME

vectorstore_names = ["accounts", "documentation", "logistics", "manager", "operators", "other", "products", "supervisors", "webmasrers"]

db_dict = {}
for name in vectorstore_names:
    db_dict[f"{name}_db"] = Chroma(embedding_function=embeddings, persist_directory=f"./{name}",
                     collection_name=f"{name}_collection")

retriever_dict = {}
for name in vectorstore_names:
    retriever_dict[f"{name}_retriever"] = db_dict[f"{name}_db"].as_retriever()


qa_dict = {}
for name in vectorstore_names:
    qa_dict[f"{name}_qa"] = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_dict[f"{name}_retriever"])


# accounts_db = Chroma(embedding_function=embeddings, persist_directory="./accounts",
#                      collection_name="accounts_collection")
# # accounts_info = VectorStoreInfo(name="accounts", description="information useful to account-managers",
# #                                 vectorstore=accounts_db)
#
# documentation_db = Chroma(embedding_function=embeddings, persist_directory="./documentation",
#                           collection_name="documentation_collection")
# # documentation_info = VectorStoreInfo(name="documentation", description="information useful to developers",
# #                                      vectorstore=documentation_db)
#
# logistics_db = Chroma(embedding_function=embeddings, persist_directory="./logistics",
#                       collection_name="logistics_collection")
# # logistics_info = VectorStoreInfo(name="logistics", description="information useful to logistics",
# #                                  vectorstore=logistics_db)
#
# manager_db = Chroma(embedding_function=embeddings, persist_directory="./manager", collection_name="manager_collection")
# # manager_info = VectorStoreInfo(name="manager", description="information useful to managers", vectorstore=manager_db)
#
# operators_db = Chroma(embedding_function=embeddings, persist_directory="./operators",
#                       collection_name="operators_collection")
# # operators_info = VectorStoreInfo(name="operators", description="information useful to operators",
# #                                  vectorstore=operators_db)
#
# other_db = Chroma(embedding_function=embeddings, persist_directory="./other", collection_name="other_collection")
# # other_info = VectorStoreInfo(name="other", description="other info about anything", vectorstore=other_db)
#
# products_db = Chroma(embedding_function=embeddings, persist_directory="./products",
#                      collection_name="products_collection")
# # products_info = VectorStoreInfo(name="products", description="information about products", vectorstore=products_db)
#
# supervisors_db = Chroma(embedding_function=embeddings, persist_directory="./supervisors",
#                         collection_name="supervisors_collection")
# # supervisors_info = VectorStoreInfo(name="supervisors", description="information useful to supervisors",
# #                                    vectorstore=supervisors_db)
#
# webmasrers_db = Chroma(embedding_function=embeddings, persist_directory="./webmasrers",
#                        collection_name="webmasrers_collection")


# webmasrers_info = VectorStoreInfo(name="webmasrers", description="information useful to webmasrers",
#                                   vectorstore=webmasrers_db)

# router_toolkit = VectorStoreRouterToolkit(
#     vectorstores=[accounts_info, documentation_info, logistics_info, manager_info, operators_info, other_info,
#                   products_info, supervisors_info, webmasrers_info], llm=llm)
# agent_executor = create_vectorstore_router_agent(llm=llm, toolkit=router_toolkit, verbose=True)

# db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME)
# accounts_retriever = accounts_db.as_retriever()
# documentation_retriever = documentation_db.as_retriever()
# logistics_retriever = logistics_db.as_retriever()
# manager_retriever = manager_db.as_retriever()
# operators_retriever = operators_db.as_retriever()
# other_retriever = other_db.as_retriever()
# products_retriever = products_db.as_retriever()
# supervisors_retriever = supervisors_db.as_retriever()
# webmasrers_retriever = webmasrers_db.as_retriever()

# accounts_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=accounts_retriever)
# documentation_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=documentation_retriever)
# logistics_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=logistics_retriever)
# manager_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=manager_retriever)
# operators_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=operators_retriever)
# other_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=other_retriever)
# products_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=products_retriever)
# supervisors_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=supervisors_retriever)
# webmasrers_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=webmasrers_retriever)




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

        main_ingest(f"{file_path}", "./vectorstore", "zendesk_collection")
        global db, retriever, qa
        db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        shutil.rmtree(file_path)
        # return db

    elif action == 'second':
        # Обработка при выборе второй категории
        print("Collection 2 chosen")

        # Удаление кнопок из сообщения модератора
        client.edit_message_reply_markup('freeeeeet', message_id=callback_query.message.id, reply_markup=None)
        client.send_message('freeeeeet', "Вы выбрали 'Категория 2'")

        main_ingest(f"./files/{file_name[:-4]}", "./vectorstore_2", "zendesk_collection_2")
        # db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME)
        shutil.rmtree(f"./files/{file_name[:-4]}")


@client.on_message(filters.command(["start"], prefixes=["/", "!"]))
async def start(client, message):
    await message.reply_text(f"""Здравствуйте, {message.from_user.first_name}! Предлагаем вам опробовать умного бота на\
 основе GPT для наших операторов и других сотрудниках примере нашей базы знаний из Zendesk. Бот хранит в памяти 4\
 последних сообщения и дает ответс учётом этих данных. Историю сообщений можно очистить с помощью команды\
 /clear_chat_history\n
Также, бот умеет искать информацию в интернете, для этого начните ваш вопрос со слов \"Найди в интернете\"""")


@client.on_message(filters.command(["clear_chat_history"], prefixes=["/", "!"]))
async def start(client, message):
    file_path = f"{CHAT_HISTORY_PATH}/chat_id_{message.chat.id}.json"
    os.remove(file_path)
    await message.reply_text(f"""{message.from_user.first_name}, ваша история сообщений с ботом очищена!""")


@client.on_message(filters.command(["help"], prefixes=["/", "!"]))
async def start(client, message):
    await message.reply_text(f"""Список доступных команд:\n
/start : Приветственное сообщение\n
/clear_chat_history : Очистить историю чата\n
\"Найди в интернете\" : Если добавить эту фразу в начале вашего вопроса, то бот будет искать информацию в интернете""")


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
    global answer_dict, final_answer
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
        if "Найди в интернете" in query:
            search_result = search.run(query.replace("Найди в интернете", ""))
            print(search_result)
            answer = llm(f"Дай краткую выжимку из этого текста: {search_result}")
        else:
            if chat_history != "":
                # answer = agent_executor.run(f"Ты оператор колл-центра. Предыдущий диалог с клиентом представлен ниже. Тебе нужно его продолжить. Ответ необходимо дать на русском языке\n{chat_history}\nКлиент:{query}\nОператор:")
                # answer = qa.run(
                #     # f"Give a detailed answer to that question: {query} \nThe answer should be in the same language as the question."
                #     # f"Ты оператор колл-центра с десятилетним стажем. У тебя есть база данных с ответов на вопросы клиентов с рекомендациями, которую ты используешь для ответа клиенту. Ты должен выбрать десять ответов и на их основе создать новый ответ, а также, учесть все замечания из рекомендаций. Твоя основная задача - продать клиенту продукт всеми доступными способами, желательно, с одного ответа на вопрос. Вопрос: {query}"
                #     # f"Ты тренер колл-центра по продажам, который выдает лучшие качественные и детальные рекомендации по продажам продукта. Вопрос: {query}"
                #     f"Ты оператор колл-центра. Предыдущий диалог с клиентом представлен ниже. Тебе нужно его продолжить.\n{chat_history}\nКлиент:{query}\nОператор:"
                # )
                answer_dict = {}
                for name in vectorstore_names:
                    answer_dict[f"{name}_answer"] = qa_dict[f"{name}_qa"].run(
                        f"Ты оператор базы знаний, который предоставляет информацию о продуктах компании, занимающихся продажей био-активных добавок. Вот предыдущий диалог с клиентом:\n{chat_history}\nКлиент:{query}\nОператор:"
                    )
                    answer = answer_dict[f"{name}_answer"]
                    print(f"Ответ от {name}:\n {answer}")

                    answers = ""
                    for key, value in answer_dict.items():
                        formatted_answer = f"{key.capitalize().replace('_', ' ')}: {value}\n\n"
                        answers += formatted_answer
                    final_answer = llm(
                        f"Ты оператор базы знаний, который предоставляет информацию о продуктах компании, занимающихся продажей био-активных добавок. Вот предыдущий диалог с клиентом:\n{chat_history}\nКлиент:{query}\n\n Твоя задача выдать максимально полный, не противоречивый ответ. Данные, которые ты получил:\n{answers}")


            else:
                answer_dict = {}
                for name in vectorstore_names:
                    answer_dict[f"{name}_answer"] = qa_dict[f"{name}_qa"].run(query)
                    answer = answer_dict[f"{name}_answer"]
                    print(f"Ответ от {name}:\n {answer}")

                    answers = ""
                    for key, value in answer_dict.items():
                        formatted_answer = f"{key.capitalize().replace('_', ' ')}: {value}\n\n"
                        answers += formatted_answer
                    final_answer = llm(
                        f"Вопрос: {query}\n Выбери лучший ответ из представленных и напиши его мне. Ответы:\n{answers}")

                # print(answer_dict)
                # answer = agent_executor.run(f"Ответь мне на русском языке: {query}")
                # answer = qa.run(query)


        new_entry = {
            "client_message": f"{query}",
            # "operator_message": f"Пока что тут не будет особо сообщений"
            "operator_message": f"{final_answer}"
        }
        # print(answer)
        data.append(new_entry)

        if len(data) >= 5:
            data.pop(0)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        # ЗАПИСАЛИ CHAT_HISTORY

        message.reply_text(final_answer)
        answer_dict = {}

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
