import asyncio
import os
import platform
from collections import Counter, defaultdict

import psycopg2
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from gigachat.exceptions import ResponseError
from langchain.chains import ConversationChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import GigaChat
from langchain_community.embeddings import GigaChatEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from config import bot_token, connection_params, sber
from langchain.callbacks.base import BaseCallbackHandler


class FinishReasonCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.finish_reason = None

    def on_llm_end(self, response, **kwargs):
        chat_generation = response.generations[0][0]
        self.finish_reason = chat_generation.message.response_metadata.get("finish_reason")
        print("Finish reason:", self.finish_reason)


# Define states for FSM
class CategorySelection(StatesGroup):
    waiting_for_category = State()
    waiting_for_question = State()


# Global data (consider using a database for persistent storage)
user_conversations = {}
user_llm_rag = {}
user_context_info = {}

# Prompts
# qa_system_prompt = """Please read the context provided below:
# CONTEXT
#
# {context}
#
# Based solely on the information given in the context above, answer the following question. If the information isn’t available in the context to formulate an answer, simply reply with ‘NO_ANSWER’. Please do not provide additional explanations or information.
#
# Question: {input}"""

# qa_system_prompt_example = '''<s> [INST] Твоя роль – консультировать клиентов компании «Первая сваяная компания» по вопросам покупки свай в Иркутске и прилегающих районах. Придерживайся делового стиля общения, будь приветливым и профессиональным.
# 1. Если клиент задает вопрос о сваях или услугах компании, предоставь актуальные цены на интересующий товар или услугу.
# 2. Если клиент упоминает свой участок:
#    - Спроси: «Пожалуйста, уточните, где расположен ваш участок (населенный пункт или район)?»
#    - После получения информации о местоположении спроси: «Для планирования выезда инженера на участок, пожалуйста, оставьте ваш номер телефона.»
# 3. Если запрос клиента не может быть выполнен (например, информация отсутствует или услуга недоступна), сообщи об этом кратко и вежливо.
# Лимит сообщения: 300 символов. Используй только информацию из контекста.
# Будь внимателен к деталям и четко следуй инструкциям. </s> [/INST]
# Вопрос: {input}
# Контекст: {context}'''

# qa_system_prompt = '''<s> [INST] Твоя роль – отвечать на вопросы по информационной, компьютерной и кибербезопасностью.
# Лимит сообщения: 300 символов. Используй только информацию из контекста. В ответе никогда не пиши слово контекст.
# Будь внимателен к деталям и четко следуй инструкциям. </s> [/INST]
# Вопрос: {input}
# Контекст: {context}'''

# qa_system_prompt = '''<s> [INST] Твоя роль — отвечать на вопросы в области информационной, компьютерной и кибербезопасности.
# Лимит сообщения: 300 символов. Используй только информацию, предоставленную в запросе, если не знаешь ответ на вопрос просто выведи NO_ANSWER. Никогда не используй слово "контекст" или любые его производные (например, "на основе контекста", "из контекста" и т.д.).
# Будь внимателен к деталям, четко следуй инструкциям и формулируй ответы так, как будто информация исходит от тебя. </s> [/INST]
# Вопрос: {input}
# Контекст: {context}'''

qa_system_prompt = '''<s> [INST] Твоя роль — отвечать на вопросы в области информационной, компьютерной и кибербезопасности. 
Лимит сообщения: 300 символов. Используй только информацию, предоставленную в запросе. Если ты не знаешь ответ на вопрос или информация отсутствует, строго выведи NO_ANSWER и ничего больше. 
Никогда не используй слово "контекст" или любые его производные (например, "на основе контекста", "из контекста" и т.д.). Будь внимателен к деталям, четко следуй инструкциям и формулируй ответы так, как будто информация исходит от тебя. </s> [/INST]
Вопрос: {input} 
Контекст: {context}'''


# qa_system_prompt = 'You can only make conversations based on the provided context. If a response cannot be formed strictly using the context, politely say you don’t have knowledge about that topic. Контекст: {context} Вопрос: {input}'

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = """Учитывая историю чата и последний вопрос пользователя, \
который может ссылаться на контекст в истории чата, сформулируйте самостоятельный вопрос, \
который можно понять без истории чата. Не отвечайте на вопрос, \
просто переформулируйте его при необходимости, а в противном случае верните его как есть."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Initialize bot and dispatcher
bot = Bot(token=bot_token)
dp = Dispatcher()




# Helper functions
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


def create_rag_chain(llm, embedding_retriever, qa_prompt):
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
    return conversational_rag_chain


def create_conversation_chain(user_id, llm):
    conversation = ConversationChain(llm=llm,
                                     verbose=True,
                                     memory=ConversationBufferMemory())

    return conversation

category_to_path = {
        "Нормативные акты ФСТЭК": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_fstek_cosine",
        "Приказы ФСБ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_fsb_cos",
        "Федеральные законы": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_fz_cosine",
        "Национальные стандарты РФ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_prikaz_cb_rf_cosine",
        "Приказы ЦБ РФ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_prikaz_cb_rf_cosine",
        "Постановления Правительства РФ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_statement_gov_rf_cosine",
        "Указы Президента РФ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_ukazy_prez_cosine",
        "Общие вопросы по ИБ": "/Users/gd/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_kaspersky_encycl_cos",
    }



# def create_llm_rag(user_id, category=None):
#     llm = GigaChat(credentials=sber,
#                    model='GigaChat-2:latest',
#                    verify_ssl_certs=False,
#                    profanity_check=False,
#                    )
#
#     llm_checker = GigaChat(credentials=sber,
#                            model='GigaChat-2-Max:latest',
#                            verify_ssl_certs=False,
#                            profanity_check=False,
#                            )
#
#     embeddings = GigaChatEmbeddings(credentials=sber, verify_ssl_certs=False)
#
#     if not category:
#         user_paths = {
#             'nikitacesev': "/Users/nikitacesev/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_3",
#             'gd': "/Users/gd/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_cleaned_3"
#         }
#         current_user = os.getenv('USER')
#         chromadb_path = user_paths.get(current_user,
#                                        "/Users/nikitacesev/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_3")
#     chromadb_path = category_to_path[category]
#     vector_store = Chroma(
#         persist_directory=chromadb_path,
#         embedding_function=embeddings
#     )
#     print(vector_store._collection_metadata)
#
#     print(chromadb_path)
#     print(len(vector_store._collection.get(include=["documents", "metadatas", "embeddings"])['documents']))
#
#
#     retriever = vector_store.as_retriever(
#         search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 4 }
#     )
#
#
#     # retriever = vector_store.as_retriever(
#     #     search_kwargs={"k": 16 }
#     # )
#
#     history_aware_retriever = create_history_aware_retriever(
#         llm, retriever, contextualize_q_prompt
#     )
#
#     # Пример
#     # 1:
#     # Вопрос: "Как защитить данные от утечек?"
#     # Категория: "Кибербезопасность"
#     # Ответ: да
#     #
#     # Пример
#     # 2:
#     # Вопрос: "Как настроить Wi-Fi роутер?"
#     # Категория: "Кибербезопасность"
#     # Ответ: нет
#     # < / s > [ / INST]
#
#     prompt = f'''<s> [INST] Относится ли заданный вопрос к категории "{category}"?
#     Категория "{category}" включает вопросы, связанные с "{category}", их содержанием, количеством и применением.
#     Учти историю запросов пользователя и контекст. Отвечай только "да" или "нет".
#     </s> [/INST]
#
#     Вопрос: {{input}}
#     Контекст: {{context}}'''
#     print(prompt)
#
#     # prompt = f'''<s> [INST] Относится ли заданный вопрос к категории "{category}"?
#     #             Учти историю запросов пользователя и контекст. Отвечай только "да" или "нет". </s> [/INST]
#     #
#     #             Вопрос: {{input}}
#     #             Контекст: {{context}}'''
#                 # Твой ответ:'''
#
#
#     qa_prompt_checker = ChatPromptTemplate.from_messages(
#         [
#             ("system", prompt),
#             MessagesPlaceholder(variable_name="chat_history", n_messages=5),
#             ("human", "{input}"),
#         ]
#     )
#
#     rag_chain = create_rag_chain(llm, history_aware_retriever, qa_prompt)
#     rag_chain_checker = create_rag_chain(llm_checker, history_aware_retriever,qa_prompt_checker )
#
#     conversation_chain = create_conversation_chain(user_id, llm)
#     return vector_store, history_aware_retriever, llm, rag_chain, rag_chain_checker, conversation_chain

def create_llm_rag(user_id, category=None):
    llm = GigaChat(credentials=sber,
                   model='GigaChat-2:latest',
                   verify_ssl_certs=False,
                   profanity_check=False,
                   
                   )

    llm_checker = GigaChat(credentials=sber,
                           model='GigaChat-2-Max:latest',
                           verify_ssl_certs=False,
                           profanity_check=False,
                           )

    embeddings = GigaChatEmbeddings(credentials=sber, verify_ssl_certs=False)

    if not category:
        user_paths = {
            'nikitacesev': "/Users/nikitacesev/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_3",
            'gd': "/Users/gd/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_cleaned_3"
        }
        current_user = os.getenv('USER')
        chromadb_path = user_paths.get(current_user,
                                       "/Users/nikitacesev/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_3")
    else:
        chromadb_path = category_to_path[category]

    vector_store = Chroma(
        persist_directory=chromadb_path,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 4}
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Generate the dynamic prompt for the checker
    if category:
        prompt = f'''<s> [INST] Относится ли заданный вопрос к категории "{category}"? 
        Категория "{category}" включает вопросы, связанные с "{category}", их содержанием, количеством и применением.
        Учти историю запросов пользователя и контекст. Отвечай только "да" или "нет". 
        </s> [/INST]

        Вопрос: {{input}} 
        Контекст: {{context}}'''
    else:
        prompt = qa_system_prompt  # Default prompt if no category is selected

    qa_prompt_checker = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "{input}"),
        ]
    )

    rag_chain = create_rag_chain(llm, history_aware_retriever, qa_prompt)
    rag_chain_checker = create_rag_chain(llm_checker, history_aware_retriever, qa_prompt_checker)

    conversation_chain = create_conversation_chain(user_id, llm)
    return vector_store, history_aware_retriever, llm, rag_chain, rag_chain_checker, conversation_chain




@dp.message(Command("start"))
async def start_command(message: types.Message):
    user_id = message.chat.id

    if user_id not in user_conversations:
        user_conversations[user_id] = ChatMessageHistory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True,
                                 keyboard=[[KeyboardButton("Очистить историю")]])

    await message.answer('Готов к работе', reply_markup=markup)


@dp.message(F.text.lower() == "очистить историю")
async def clear_history_button(message: types.Message):
    user_id = message.chat.id

    if user_id in user_conversations:
        user_conversations[user_id].clear()
        await message.answer("История сообщений очищена.", reply_markup=ReplyKeyboardRemove())
    else:
        await message.answer("История сообщений уже пуста.", reply_markup=ReplyKeyboardRemove())


@dp.message(Command("help"))
async def help_command(message: types.Message):
    await message.answer('Я - бот-помощник, отвечающий на вопросы по теме информационной безопасности.')


@dp.message(Command("clear"))
async def clear_history_command(message: types.Message):
    user_id = message.chat.id

    if user_id in user_conversations:
        user_conversations[user_id].clear()
        await message.answer("История сообщений очищена.")
    else:
        await message.answer("История сообщений уже пуста.")



@dp.message(Command("categories"))
async def categories_command(message: types.Message, state: FSMContext):
    categories = [
        "Нормативные акты ФСТЭК",
        "Приказы ФСБ",
        "Федеральные законы",
        "Национальные стандарты РФ",
        "Приказы ЦБ РФ",
        "Постановления Правительства РФ",
        "Указы Президента РФ",
        "Общие вопросы по ИБ",
    ]

    # Create a list of KeyboardButton objects
    buttons = [KeyboardButton(text=category) for category in categories]

    # Create the ReplyKeyboardMarkup with the buttons
    markup = ReplyKeyboardMarkup(
        keyboard=[buttons[i:i + 4] for i in range(0, len(buttons), 4)],  # Split into rows of 4 buttons
        resize_keyboard=True,
        one_time_keyboard=True
    )

    await state.set_state(CategorySelection.waiting_for_category)
    await message.answer("Выберите категорию базы данных:", reply_markup=markup)



# @dp.message(CategorySelection.waiting_for_category)
# async def handle_category_selection(message: types.Message, state: FSMContext):
#     selected_category = message.text
#
#     # Save the selected category in the state
#     await state.update_data(selected_category=selected_category)
#
#     # Initialize the RAG chain for the selected category
#     user_id = message.chat.id
#     user_llm_rag[user_id] = create_llm_rag(user_id, selected_category)
#
#     # Move to the next state to wait for the user's question
#     await state.set_state(CategorySelection.waiting_for_question)
#     await message.answer(f"Вы выбрали категорию: {selected_category}. Теперь задайте ваш вопрос:")
@dp.message(CategorySelection.waiting_for_category)
async def handle_category_selection(message: types.Message, state: FSMContext):
    selected_category = message.text

    # Save the selected category in the state
    await state.update_data(selected_category=selected_category)

    # Initialize the RAG chain for the selected category
    user_id = message.chat.id
    user_llm_rag[user_id] = create_llm_rag(user_id, selected_category)
    print(selected_category)

    # Move to the next state to wait for the user's question
    await state.set_state(CategorySelection.waiting_for_question)
    await message.answer(f"Вы выбрали категорию: {selected_category}. Теперь задайте ваш вопрос:")


# @dp.message(StateFilter(CategorySelection.waiting_for_question), F.text)
# async def handle_text_message(message: types.Message, state: FSMContext):
#     user_id = message.chat.id
#
#     # Initialize user conversation and RAG chain if not already done
#     if user_id not in user_conversations:
#         user_conversations[user_id] = ChatMessageHistory()
#
#     if user_id not in user_llm_rag:
#         user_llm_rag[user_id] = create_llm_rag(user_id)
#
#     vdb, embedding_retriever, llm, rag_chain, rag_chain_checker, conversation = user_llm_rag[user_id]
#     conversation.memory = user_conversations[user_id]
#
#     q1 = message.text
#     print(q1)
#     print(rag_chain_checker)
#
#     try:
#         # Retrieve the selected category from the state (if any)
#         data = await state.get_data()
#         selected_category = data.get("selected_category", None)
#
#         callback_handler = FinishReasonCallbackHandler()
#         # If a category is selected, dynamically generate the prompt
#         is_question_belongs2sel_cat = True
#         if selected_category:
#
#             # prompt = f'''Относится ли вопрос "{q1}" к категории "{selected_category}"?
#             # Учти историю запросов пользователя и контекст. Отвечай только "да" или "нет".
#             #
#             # Пример 1:
#             # Вопрос: "Как защитить данные от утечек?"
#             # Категория: "Кибербезопасность"
#             # Ответ: да
#             #
#             # Пример 2:
#             # Вопрос: "Как настроить Wi-Fi роутер?"
#             # Категория: "Кибербезопасность"
#             # Ответ: нет
#             #
#             # Твой ответ:'''
#             # prompt = f'Относится ли вопрос: {q1} к {selected_category} (используй историю запросов пользователя)? Отвечай только да или нет.'
#
#             init_resp = await rag_chain_checker.ainvoke({'input': q1}, config={"callbacks": [callback_handler], 'configurable': {'session_id': user_id}})
#         #
#         #     # If the question is not related to the selected category, respond accordingly
#             print(init_resp.keys())
#             if init_resp['answer'].lower() in ['нет.', 'нет']:
#                 is_question_belongs2sel_cat = False
#                 await message.answer("Ваш вопрос не относится к выбранной категории.")
#                 return
#         print(is_question_belongs2sel_cat)
#         # Proceed with the RAG pipeline
#
#         resp = await rag_chain.ainvoke(
#             {'input': q1}, config={"callbacks": [callback_handler], 'configurable': {'session_id': user_id}}
#         )
#
#         # Reset user context info
#         user_context_info['name_docs'] = []
#         user_context_info['new_name_docs'] = []
#         user_context_info['folder_names'] = []
#         user_context_info['docs_by_folder'] = defaultdict(set)
#
#         # Handle the response based on the finish reason
#         if callback_handler.finish_reason == 'blacklist':
#             answer = "Ваш запрос не может быть обработан, так как он относится к категории, которая ограничена политикой модели"
#         elif len(resp['context']) == 0 or resp['answer'] == 'NO_ANSWER':
#             answer = 'Не основано на документах'
#         else:
#             # Extract document sources and organize them by folder
#             for i in range(1):
#                 if 'source' in resp['context'][i].metadata:
#                     source_path = resp['context'][i].metadata['source']
#                     folder_name = os.path.basename(os.path.dirname(source_path))
#                     user_context_info['docs_by_folder'][folder_name].add(
#                         os.path.splitext(os.path.basename(source_path))[0])
#
#             # Construct the answer with the response and document sources
#             answer = f"Ответ на ваш вопрос:\n\n{resp['answer']}\n\n"
#             answer += "Основано на следующих документах:\n\n"
#             for folder_name, docs in user_context_info['docs_by_folder'].items():
#                 answer += f"\nКатегория: {folder_name}\nДокументы:\n"
#                 for doc in docs:
#                     answer += f"- {doc}\n"
#
#         await message.answer(answer)
#     except ResponseError as ex:
#         print(ex)
#         await message.answer("Ваш запрос слишком длинный. Пожалуйста, сократите его и попробуйте снова.")
#     # finally:
#         # Clear the state after processing
#         # await state.clear()

@dp.message(StateFilter(CategorySelection.waiting_for_question), F.text)
async def handle_text_message(message: types.Message, state: FSMContext):
    user_id = message.chat.id

    # Initialize user conversation and RAG chain if not already done
    if user_id not in user_conversations:
        user_conversations[user_id] = ChatMessageHistory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, rag_chain_checker, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    q1 = message.text

    try:
        # Retrieve the selected category from the state (if any)
        data = await state.get_data()
        selected_category = data.get("selected_category", None)

        callback_handler = FinishReasonCallbackHandler()

        # If a category is selected, check if the question belongs to it
        is_question_belongs2sel_cat = True
        if selected_category:
            init_resp = await rag_chain_checker.ainvoke(
                {'input': q1}, config={"callbacks": [callback_handler], 'configurable': {'session_id': user_id}}
            )
            # print(rag_chain_checker.config)
            # print(rag_chain_checker.bound.__dir__())
            # print(rag_chain_checker.bound.name)
            # print(rag_chain_checker.bound.bound)
            # print(rag_chain_checker.bound.kwargs)
            # print(rag_chain_checker.bound.config)


            if init_resp['answer'].lower() in ['нет.', 'нет']:
                is_question_belongs2sel_cat = False
                await message.answer("Ваш вопрос не относится к выбранной категории.")
                return

        # Proceed with the RAG pipeline
        resp = await rag_chain.ainvoke(
            {'input': q1}, config={"callbacks": [callback_handler], 'configurable': {'session_id': user_id}}
        )
        user_context_info['name_docs'] = []
        user_context_info['new_name_docs'] = []
        user_context_info['folder_names'] = []
        user_context_info['docs_by_folder'] = defaultdict(set)

        # Handle the response
        if callback_handler.finish_reason == 'blacklist':
            answer = "Ваш запрос не может быть обработан, так как он относится к категории, которая ограничена политикой модели"
        elif len(resp['context']) == 0 or resp['answer'] == 'NO_ANSWER':
            answer = 'Не основано на документах'
        else:
            # # Construct the answer with the response and document sources
            # answer = f"Ответ на ваш вопрос:\n\n{resp['answer']}\n\n"
            # answer += "Основано на следующих документах:\n\n"
            # for folder_name, docs in user_context_info['docs_by_folder'].items():
            #     answer += f"\nКатегория: {folder_name}\nДокументы:\n"
            #     for doc in docs:
            #         answer += f"- {doc}\n"

            for i in range(1):
                if 'source' in resp['context'][i].metadata:
                    source_path = resp['context'][i].metadata['source']
                    folder_name = os.path.basename(os.path.dirname(source_path))
                    user_context_info['docs_by_folder'][folder_name].add(
                        os.path.splitext(os.path.basename(source_path))[0])

                # Construct the answer with the response and document sources
                answer = f"Ответ на ваш вопрос:\n\n{resp['answer']}\n\n"
                answer += "Основано на следующих документах:\n\n"
                for folder_name, docs in user_context_info['docs_by_folder'].items():
                    answer += f"\nКатегория: {folder_name}\nДокументы:\n"
                    for doc in docs:
                        answer += f"- {doc}\n"

        await message.answer(answer)
    except ResponseError as ex:
        print(ex)
        await message.answer("Ваш запрос слишком длинный. Пожалуйста, сократите его и попробуйте снова.")

@dp.message( F.text)
async def handle_text_message(message: types.Message, state: FSMContext):
    await message.answer("Пожалуйста, выберите категорию базы данных с помощью команды /categories.")
@dp.message((F.content_type.in_({types.ContentType.AUDIO, types.ContentType.VIDEO,
                                 types.ContentType.PHOTO, types.ContentType.STICKER,
                                 types.ContentType.VOICE, types.ContentType.LOCATION,
                                 types.ContentType.CONTACT})))
async def not_text_message(message: types.Message):
    user_id = message.chat.id
    await message.answer('Я работаю только с текстовыми сообщениями!')


# Main function to start the bot
async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())