import os
import json

def format_text(text_data):
    if isinstance(text_data, list):
        # Если текст представлен в виде списка объектов, объединяем его
        formatted_text = ""
        for item in text_data:
            if "text" in item:
                formatted_text += item["text"]
    elif isinstance(text_data, str):
        formatted_text = text_data
    else:
        formatted_text = ""

    return formatted_text

# Путь к JSON-файлу
json_file = "result.json"

# Проверяем, существует ли файл
if not os.path.exists(json_file):
    print(f"Файл {json_file} не найден.")
else:
    # Открываем JSON-файл и загружаем данные
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    output_directory = "message_files"

    # Создаем каталог для хранения файлов сообщений, если его еще нет
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Итерируемся по сообщениям и создаем файлы
    for message in data["messages"]:
        if message["type"] == "message":
            message_date = message["date"].split("T")[0]
            message_text = format_text(message["text"])

            if message_text:

                # Создаем имя файла на основе даты сообщения
                filename = os.path.join(output_directory, f"{message_date}_{message['id']}.txt")

                # Записываем данные в файл
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(f"Данные актуальны на {message_date}\n\n")
                    file.write(message_text)

    print("Созданы текстовые файлы для сообщений.")