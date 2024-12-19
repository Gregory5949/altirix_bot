import os
import platform

from gigachat.exceptions import ResponseError
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import GigaChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.embeddings import GigaChatEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnableWithMessageHistory

from config import sber, bot_token, connection_params

user_conversations = {}
user_llm_rag = {}

qa_system_prompt = '''Ты чат-бот помощник, созданный компанией Альтирикс системс. You MUST answer to questions only in Russian. Вы отвечаете только на вопросы по темам информационной безопасности, компьютерной безопасности и кибербезопасности. Если вопрос задан на другом языке или в вашем хранилище RAG нет информации для ответа, сообщите об этом пользователю. Не отвечайте на общие вопросы, такие как "Как дела?" и вопросы, заданные не на русском языке. Если вопрос связан с нормативными документами, необходимо привести фрагмент из опредленного нормативного документа, обосновывающего ответ. Ответ должен быть до 200 символов. Прежде чем ответить, проанализируйте запрос и определите его тематическую область.
Контекст: {context}
Вопрос: {input}'''

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
import telebot
from time import sleep
from telebot import types

bot = telebot.TeleBot(bot_token)

contextualize_q_system_prompt = """Учитывая историю чата и последний вопрос пользователя, \
который может ссылаться на контекст в истории чата, сформулируйте самостоятельный вопрос, \
который можно понять без истории чата. Не отвечайте на вопрос, \
просто переформулируйте его при необходимости, а в противном случае верните его как есть."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
from langchain_community.chat_message_histories import ChatMessageHistory
import psycopg2


def check_rate_limit(user_id, user_message):
    with psycopg2.connect(**connection_params) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM request_logs 
                WHERE user_id = %s AND created_at >= NOW() - INTERVAL '1 day';
            """, (user_id,))

            count = cursor.fetchone()[0]

            if count >= 10:
                return False
            else:
                cursor.execute("INSERT INTO request_logs (user_id, user_message) VALUES (%s, %s);",
                               (user_id, user_message))
                connection.commit()
                return True


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in user_conversations:
        user_conversations[session_id] = ChatMessageHistory()
    return user_conversations[session_id]


def create_llm_rag(user_id):
    llm = GigaChat(credentials=sber,
                   model='GigaChat:latest',
                   verify_ssl_certs=False,
                   profanity_check=False,
                   )

    embeddings = GigaChatEmbeddings(credentials=sber, verify_ssl_certs=False)
    chromadb_path = "./chromadb_chunk_size_1200"
    vector_store = Chroma(
        persist_directory=chromadb_path,
        embedding_function=embeddings)
    retriever_vanilla = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 16, })
    retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 16, })

    print(len(vector_store.get()['documents']))
    retriever_BM25 = BM25Retriever.from_texts(vector_store.get()['documents'])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.3, 0.3, 0.4]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )

    rag_chain = create_rag_chain(llm, history_aware_retriever)

    conversation_chain = create_conversation_chain(user_id, llm)
    return (vector_store, history_aware_retriever, llm, rag_chain, conversation_chain)


def create_rag_chain(llm, embedding_retriever):
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )
    retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return (conversational_rag_chain)


def create_conversation_chain(user_id, llm):
    conversation = ConversationChain(llm=llm,
                                     verbose=True,
                                     memory=ConversationBufferMemory())

    return (conversation)


@bot.message_handler(commands=['start'])
def start(message: types.Message):
    user_id = message.chat.id

    if user_id not in user_conversations:
        user_conversations[user_id] = ChatMessageHistory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    bot.send_message(message.chat.id, 'Готов к работе')


@bot.message_handler(commands=['help'])
def help(message: types.Message):
    bot.send_message(message.chat.id,
                     'Я - бот-помощник, отвечающий на вопросы по теме информационной безопасности.')


@bot.message_handler(content_types=['audio',
                                    'video',
                                    'photo',
                                    'sticker',
                                    'voice',
                                    'location',
                                    'contact'])
def not_text(message):
    user_id = message.chat.id
    bot.send_message(user_id, 'Я работаю только с текстовыми сообщениями!')


@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    user_id = message.chat.id

    if user_id not in user_conversations:
        user_conversations[user_id] = ChatMessageHistory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    q1 = message.text

    try:
        if check_rate_limit(user_id, q1):
            resp1 = rag_chain.invoke(
                {'input': q1}, config={'configurable': {'session_id': user_id}}
            )

            answer = resp1['answer']
            bot.send_message(user_id, answer)
        else:
            bot.send_message(user_id, "Вы уже отправили боту 10 запросов за эти сутки. ")
    except ResponseError:
        bot.send_message(user_id, "Ваш запрос слишком длинный. Пожалуйста, сократите его и попробуйте снова.")

    sleep(2)


bot.polling(none_stop=True)
