import asyncio
import os
from collections import defaultdict
import nltk
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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import GigaChat
from langchain_community.embeddings import GigaChatEmbeddings
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from nltk import SnowballStemmer

from config import bot_token, connection_params, sber
from langchain.callbacks.base import BaseCallbackHandler
import yake


class FinishReasonCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.finish_reason = None

    def on_llm_end(self, response, **kwargs):
        chat_generation = response.generations[0][0]
        self.finish_reason = chat_generation.message.response_metadata.get("finish_reason")
        print("Finish reason:", self.finish_reason)


class CategorySelection(StatesGroup):
    waiting_for_category = State()
    waiting_for_question = State()


user_conversations = {}
user_llm_rag = {}
user_context_info = {}

qa_system_prompt = '''<s> [INST] Твоя роль — отвечать на вопросы в области информационной, компьютерной и кибербезопасности, используя точные формулировки из предоставленного контекста. 
Лимит ответа: 300 символов. Отвечай кратко. Если ты не знаешь ответ на вопрос или информация отсутствует, выведи NO_ANSWER и ничего больше. 
Никогда не используй слово "контекст" или любые его производные (например, "на основе контекста", "из контекста" и т.д.). Будь внимателен к деталям, четко следуй инструкциям и формулируй ответы так, как будто информация исходит от тебя. </s> [/INST]
Вопрос: {input} 
Контекст: {context}
'''

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

bot = Bot(token=bot_token)
dp = Dispatcher()


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


def create_conversation_chain(llm):
    conversation = ConversationChain(llm=llm,
                                     verbose=True,
                                     memory=ConversationBufferMemory())

    return conversation


# было: /Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_fstek_cosine и 1900, щас 2000
# было: /Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_statement_gov_rf_cosine и 1900, щас 2100
# было: /Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_fz_cosine (в 1900 версии, файл Сравнение соглашений о поручении на обработку персональных данных и о передаче персональных данных - RPPA.pdf – 1700 чанк
# было: /Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_ukazy_prez_cosine, щас 2300
# было /Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_fsb_cos, щас 2200
# добавидась /Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_2200_roskomnadzor_cosine
# убрана "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1200_prikaz_cb_rf_cosine"

category_to_path = {
    "Приказы ФСТЭК": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_2000_fstek_cosine",
    "Приказы ФСБ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_2200_fsb_cosine",
    "Федеральные законы": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_1900_fz_cosine",
    "Приказы Роскомнадзора": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_2200_roskomnadzor_cosine",
    "Постановления Правительства РФ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_2100_statement_gov_rf_cosine",
    "Указы Президента РФ": "/Users/gd/PycharmProjects/altirix_systems_chatbot/chromadb_chunk_size_2300_ukazy_prez_cosine",
    "Общие вопросы по ИБ": "/Users/gd/PycharmProjects/altirix_bot/LitPom_bot-main/chromadb_chunk_size_1200_kaspersky_encycl_cos",
}


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
                           temperature=0
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
    print(len(vector_store._collection.get()['documents']))

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 16}
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    if category:

        prompt = f'''<s> [INST] Относится ли заданный запрос к категории {category}?
                Отвечай только "да" или "нет".
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
    print(prompt)

    rag_chain = create_rag_chain(llm, history_aware_retriever, qa_prompt)
    rag_chain_checker = create_rag_chain(llm_checker, history_aware_retriever, qa_prompt_checker)

    conversation_chain = create_conversation_chain(llm)
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
        "Приказы ФСТЭК",
        "Приказы ФСБ",
        "Федеральные законы",
        "Приказы Роскомнадзора",
        "Постановления Правительства РФ",
        "Указы Президента РФ",
        "Общие вопросы по ИБ",
    ]

    buttons = [KeyboardButton(text=category) for category in categories]

    markup = ReplyKeyboardMarkup(
        keyboard=[buttons[i:i + 4] for i in range(0, len(buttons), 4)],  # Split into rows of 4 buttons
        resize_keyboard=True,
        one_time_keyboard=True
    )

    await state.set_state(CategorySelection.waiting_for_category)
    await message.answer("Выберите категорию базы данных:", reply_markup=markup)


@dp.message(CategorySelection.waiting_for_category)
async def handle_category_selection(message: types.Message, state: FSMContext):
    selected_category = message.text

    await state.update_data(selected_category=selected_category)

    user_id = message.chat.id
    user_llm_rag[user_id] = create_llm_rag(user_id, selected_category)
    print(selected_category)

    await state.set_state(CategorySelection.waiting_for_question)
    await message.answer(f"Вы выбрали категорию: {selected_category}. Теперь задайте ваш вопрос:")


@dp.message(StateFilter(CategorySelection.waiting_for_question), F.text)
async def handle_text_message(message: types.Message, state: FSMContext):
    user_id = message.chat.id

    if user_id not in user_conversations:
        user_conversations[user_id] = ChatMessageHistory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, rag_chain_checker, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    q1 = message.text

    from nltk.corpus import stopwords
    nltk.download('stopwords')




    try:
        data = await state.get_data()
        selected_category = data.get("selected_category", None)

        callback_handler = FinishReasonCallbackHandler()

        if selected_category:
            init_resp = await rag_chain_checker.ainvoke(
                {'input': q1},
                config={"callbacks": [callback_handler], 'configurable': {'session_id': user_id}}
            )

            if init_resp['answer'].lower() in ['нет.', 'нет']:
                await message.answer("Ваш вопрос не относится к выбранной категории.")
                return

        resp = await rag_chain.ainvoke(
            {'input': q1}, config={"callbacks": [callback_handler], 'configurable': {'session_id': user_id}}
        )
        user_context_info['name_docs'] = []
        user_context_info['new_name_docs'] = []
        user_context_info['folder_names'] = []
        user_context_info['docs_by_folder'] = defaultdict(set)

        if callback_handler.finish_reason == 'blacklist':
            answer = "Ваш запрос не может быть обработан, так как он относится к категории, которая ограничена политикой модели"
        elif len(resp['context']) == 0 or resp['answer'] == 'NO_ANSWER':
            answer = 'Не основано на документах'
        else:
            for i in range(1):
                if 'source' in resp['context'][i].metadata:
                    source_path = resp['context'][i].metadata['source']
                    folder_name = os.path.basename(os.path.dirname(source_path))
                    user_context_info['docs_by_folder'][folder_name].add(
                        os.path.splitext(os.path.basename(source_path))[0])
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


@dp.message(F.text)
async def handle_text_message(message: types.Message, state: FSMContext):
    await message.answer("Пожалуйста, выберите категорию базы данных с помощью команды /categories.")


@dp.message((F.content_type.in_({types.ContentType.AUDIO, types.ContentType.VIDEO,
                                 types.ContentType.PHOTO, types.ContentType.STICKER,
                                 types.ContentType.VOICE, types.ContentType.LOCATION,
                                 types.ContentType.CONTACT})))
async def not_text_message(message: types.Message):
    user_id = message.chat.id
    await message.answer('Я работаю только с текстовыми сообщениями!')


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
