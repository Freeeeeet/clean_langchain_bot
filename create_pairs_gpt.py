import os
from dotenv import load_dotenv

from langchain.llms import OpenAI, HuggingFacePipeline

load_dotenv()

llm = OpenAI(temperature=0.3,
             # model_name="gpt-3.5-turbo-16k"
             )

# Путь к папке с файлами сообщений
message_directory = "message_files"

# Проверяем, существует ли папка
if not os.path.exists(message_directory):
    print(f"Папка {message_directory} не найдена.")
else:
    # Итерируемся по файлам в папке
    for filename in os.listdir(message_directory):
        # Полный путь к текущему файлу
        file_path = os.path.join(message_directory, filename)

        # Генерируем новое имя файла с префиксом "message_"
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Пропускаем первые две строки (актуальность и дата), и выводим оставшийся текст
        message_text = "".join(lines[2:])

        situation_reaction = llm(
            f""""This is the response of a supervisor in a call center to a certain situation: {message_text}
Think of this situation and produce a response in the following format:
Situation: situation description
Supervisor's response: the response described above

The entire response must be in English""")

        situation_pairs = "situation_pairs"
        if not os.path.exists(situation_pairs):
            os.makedirs(situation_pairs)
        final_path = os.path.join(situation_pairs, filename)
        with open(final_path, "w", encoding="utf-8") as final_file:
            final_file.write(situation_reaction)
        # Выводим текст на экран

        print(situation_reaction)
