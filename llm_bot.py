from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()
PROXY_API_KEY = os.getenv('PROXY_API_KEY')
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# llm от OpenAI
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=PROXY_API_KEY,
    base_url="https://api.proxyapi.ru/openai/v1"
)

template = """  
Ты бот, который помогает пользователю найти фильм, в зависимости от его предпочтений. 
Тебе нужно убедить пользователя, что именно те фильмы, которые ты ему предлагаешь ему подойдут.
Расскажи, какие чувства он испытает и что получит от просмотра каждого фильма.
Если рейтинг фильма ниже 7, то рейтинг не стоит упоминать.
В случае успеха ты получишь 100$.
Форматируй текст, используй отступы, смайлики, не используй симполы LaTeX и **.
Используй контекст.

{context}  

Вопрос: {question}

Ответ:  
"""
prompt = ChatPromptTemplate.from_template(template)

chain = RunnableSequence(
    prompt | llm
)


template_k = """  
Ты бот помошник, который ищет наиболее подходящие фильмы для пользователся.
Но для начала нужно понять, какой фильм хочет посмотреть пользователь.
Для поиска по ключевым словам выдели от 5 до 10 слов из запроса, или придумай синонимы, которые помогут найти подходящие фильмы из базы данных.

Запрос: {question}

Ключевые слова:  
"""
prompt_k = ChatPromptTemplate.from_template(template_k)

# Создаем последовательность
chain_keys = RunnableSequence(
    prompt_k | llm
)
