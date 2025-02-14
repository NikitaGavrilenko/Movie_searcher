from bot_retriever import retriever
from llm_bot import chain, chain_keys, format_docs
from dotenv import load_dotenv
import telebot
import os

# Получаем ключи
load_dotenv()
TOKEN = os.getenv('TOKEN')
bot = telebot.TeleBot(TOKEN)


# Обработка команды /start
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, '''Привет! Я бот, по подбору "того самого" фильма на вечер и вот что я умею:

    🎥 Найти фильм по жанру
    — Подберите фильм в зависимости от вашего любимого жанра!
    
    🔍 Поиск по ключевым словам
    — Укажите ключевые слова, и я найду подходящий фильм!
    
    ⭐ Рекомендовать фильм
    — Хотите рекомендацию? Я подберу что-то интересное для вас!
    
    🗓️ Новинки кино
    — Узнайте о самых последних релизах. Этот пункт поможет вам следить за новинками!
    
    📽️ Популярные фильмы
    — Посмотрите на самые популярные фильмы прямо сейчас!
    
    ❓ Ответы на вопросы
    — Есть вопросы о фильмах или жанрах? Задайте их мне!
    
    🤖 Узнать о боте
    — Узнайте больше о том, что я умею и как могу помочь!

    ''')


@bot.message_handler(content_types=["text"])
def repeat_all_messages(message):
    # Подготовка входных данных для поиска ключевых слов
    inputs_k = {
        "question": message.text
    }

    # Запуск цепочки с ключевыми словами
    response = chain_keys.invoke(inputs_k)


    # Извлечение документов из retriever
    docs = retriever.get_relevant_documents(response.content)
    context = format_docs(docs)  # формируем контекст из документов

    # Подготовка входных данных для ответа, добавления контекста
    inputs = {
        "context": context,
        "question": message.text
    }

    # Запуск основной цепочки
    response = chain.invoke(inputs)

    # Отправляем ответ пользователю
    bot.send_message(message.chat.id, response.content)


bot.polling(none_stop=True)
