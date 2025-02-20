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
from langchain_core.documents import Document
from langchain_core.runnables import chain
from collections import defaultdict
from collections import Counter
#
# import pyperclip
from langchain_community.chat_message_histories import ChatMessageHistory
import psycopg2
from config import sber, bot_token, connection_params
import telebot
from time import sleep
from telebot import types

# from langchain_deepseek import ChatDeepSeek

user_conversations = {}
user_llm_rag = {}
user_context_info = {}

# qa_system_prompt = '''Ты чат-бот помощник, созданный компанией Альтирикс системс. You MUST answer to questions only in Russian. Вы отвечаете только на вопросы по темам информационной безопасности, компьютерной безопасности и кибербезопасности. Если вопрос задан на другом языке или в вашем хранилище RAG нет информации для ответа, сообщите об этом пользователю. Не отвечайте на общие вопросы, такие как "Как дела?" и вопросы, заданные не на русском языке. Если вопрос связан с нормативными документами, необходимо привести фрагмент из опредленного нормативного документа, обосновывающего ответ. Ответ должен быть до 200 символов. Прежде чем ответить, проанализируйте запрос и определите его тематическую область.
# Контекст: {context}
# Вопрос: {input}'''

#


# qa_system_prompt = 'You can only make conversations based on the provided context. If a response cannot be formed strictly using the context, politely say you don’t have knowledge about that topic. Контекст: {context} Вопрос: {input}'

qa_system_prompt = '''Please read the context provided below:
CONTEXT

{context}

Based solely on the information given in the context above, answer the following question. If the information isn’t available in the context to formulate an answer, simply reply with ‘NO_ANSWER’. Please do not provide additional explanations or information.

Question: {input}'''

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history", n_messages=1),
        ("human", "{input}"),
    ]
)

bot = telebot.TeleBot(bot_token)

contextualize_q_system_prompt = """Учитывая историю чата и последний вопрос пользователя, \
который может ссылаться на контекст в истории чата, сформулируйте самостоятельный вопрос, \
который можно понять без истории чата. Не отвечайте на вопрос, \
просто переформулируйте его при необходимости, а в противном случае верните его как есть."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history", n_messages=1),
        ("human", "{input}"),
    ]
)

llm_checker = GigaChat(credentials=sber,
                       model='GigaChat:latest',
                       verify_ssl_certs=False,
                       profanity_check=False,
                       )


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


@bot.message_handler(commands=['clear'])
def clear_history(message: types.Message):
    user_id = message.chat.id

    if user_id in user_conversations:
        user_conversations[user_id].clear()
        bot.send_message(user_id, "История сообщений очищена.")
    else:
        bot.send_message(user_id, "История сообщений уже пуста.")


def create_llm_rag(user_id):
    llm = GigaChat(credentials=sber,
                   model='GigaChat:latest',
                   verify_ssl_certs=False,
                   profanity_check=False,
                   )
    print(llm.model)
    embeddings = GigaChatEmbeddings(credentials=sber, verify_ssl_certs=False)
    user_paths = {
        'nikitacesev': "/Users/nikitacesev/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_3",
        # 'gd': "/Users/gd/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_cleaned"
        'gd': "/Users/gd/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_cleaned_3"

    }

    current_user = os.getenv('USER')

    chromadb_path = user_paths.get(current_user,
                                   "/Users/nikitacesev/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_3")

    vector_store = Chroma(
        persist_directory=chromadb_path,
        embedding_function=embeddings)
    # retriever_vanilla = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 16, })
    # retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 16, })

    print(len(vector_store.get()['documents']))
    # retriever_BM25 = BM25Retriever.from_texts(vector_store.get()['documents'])

    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.3, 0.3, 0.4]
    # )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7,}
    )

    # history_aware_retriever = create_history_aware_retriever(
    #     llm, ensemble_retriever, contextualize_q_prompt
    # )

    rag_chain = create_rag_chain(llm, retriever)

    conversation_chain = create_conversation_chain(user_id, llm)
    return (vector_store, retriever, llm, rag_chain, conversation_chain)


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

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    clear_button = types.KeyboardButton("Очистить историю")
    markup.add(clear_button)

    bot.send_message(message.chat.id, 'Готов к работе', reply_markup=markup)


@bot.message_handler(func=lambda message: message.text.lower() == "очистить историю")
def clear_history_button(message: types.Message):
    user_id = message.chat.id

    if user_id in user_conversations:
        user_conversations[user_id].clear()
        bot.send_message(user_id, "История сообщений очищена.", reply_markup=types.ReplyKeyboardRemove())
    else:
        bot.send_message(user_id, "История сообщений уже пуста.", reply_markup=types.ReplyKeyboardRemove())


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


def calculate_file_probabilities(resp1):
    file_names = []
    for resp in resp1['context']:
        if 'source' in resp.metadata:
            file_names.append(os.path.basename(resp.metadata['source']))
    file_count = Counter(file_names)
    total_files = sum(file_count.values())
    file_probabilities = {file: count / total_files for file, count in file_count.items()}
    return file_probabilities


@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    user_id = message.chat.id

    if user_id not in user_conversations:
        user_conversations[user_id] = ChatMessageHistory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    print("Сообщений в истории сообщений: ", len(conversation.memory.messages))

    q1 = message.text

    # messages = ['Как дела?', 'Сколько будет 2+2?', 'Сколько в России городов?', 'Сколько тебе лет?']
    try:
        # if check_rate_limit(user_id, q1):
        init_resp = llm_checker.invoke(
            f'Относится ли вопрос: {q1} к информационной безопасности? Отвечай только да или нет')

        bot.send_message(user_id, init_resp.content)
        resp = rag_chain.invoke(
            {'input': q1}, config={'configurable': {'session_id': user_id}}
        )
        user_context_info['name_docs'] = []
        user_context_info['new_name_docs'] = []
        user_context_info['folder_names'] = []
        user_context_info['docs_by_folder'] = defaultdict(set)


        if resp['answer'] == 'NO_ANSWER' or len(resp['context']) == 0:
            answer = 'Не основано на документах'
        else:
            for i in range(1):
                if 'source' in resp['context'][i].metadata:
                    source_path = resp['context'][i].metadata['source']
                    # user_context_info['name_docs'].append(source_path)
                    # user_context_info['new_name_docs'].append(os.path.splitext(os.path.basename(source_path))[0])
                    folder_name = os.path.basename(os.path.dirname(source_path))
                    # user_context_info['folder_names'].append(folder_name)
                    user_context_info['docs_by_folder'][folder_name].add(
                        os.path.splitext(os.path.basename(source_path))[0])

            answer = f"Ответ на ваш вопрос:\n\n{resp['answer']}\n\n"
            answer += "Основано на следующих документах:\n\n"
            for folder_name, docs in user_context_info['docs_by_folder'].items():
                answer += f"\nКатегория: {folder_name}\nДокументы:\n"
                for doc in docs:
                    answer += f"- {doc}\n"

        bot.send_message(user_id, answer)
    #     # else:
    #     #     bot.send_message(user_id, "Вы уже отправили боту 10 запросов за эти сутки. ")
    except ResponseError as ex:
        print(ex)
        bot.send_message(user_id, "Ваш запрос слишком длинный. Пожалуйста, сократите его и попробуйте снова.")
    # # print(resp)
    # sleep(2)


bot.polling(none_stop=True)
