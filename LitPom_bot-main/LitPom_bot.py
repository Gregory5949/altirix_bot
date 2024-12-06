from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain_community.chat_models import GigaChat

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.embeddings import GigaChatEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableBranch

from config import sber, bot_token
from langgraph.graph import START, MessagesState, StateGraph

user_conversations = {}  # Словарь для хранения ConversationBufferMemory каждого пользователя
user_llm_rag = {}  # Словарь для хранения модели и rag каждого пользователя

doc_store = 'data'  # Путь для сохранения векторных хранилищ

rag_prompt = ChatPromptTemplate.from_template('''Ты чат-бот помощник, созданный компанией Альтирикс системс. You MUST answer to questions only in Russian. Вы отвечаете только на вопросы по темам информационной безопасности, компьютерной безопасности и кибербезопасности. Если вопрос задан на другом языке или в вашем хранилище RAG нет информации для ответа, сообщите об этом пользователю. Не отвечайте на общие вопросы, такие как "Как дела?" и вопросы, заданные не на русском языке. Ответ должен быть до 300 символов. Прежде чем ответить, проанализируйте запрос и определите его тематическую область.
Контекст: {context}
Вопрос: {input}''' )

# Создать объект бота
import telebot
from time import sleep
from telebot import types

bot = telebot.TeleBot(bot_token)

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": response}

def create_llm_rag(user_id):
    llm = GigaChat(credentials=sber,
                   model='GigaChat',
                   verify_ssl_certs=False,
                   profanity_check=False,

                   )

    embeddings = GigaChatEmbeddings(credentials=sber, verify_ssl_certs=False)
    vector_store = Chroma(
        persist_directory="/Users/gd/Desktop/altirix_bot_chromadbs/chroma_db_docs28885_ch_size500_ch_overlap50_gigachat_embs",
        embedding_function=embeddings)
    # print(len(vector_store._collection.get(include=['embeddings'])['embeddings']))
    # print(vector_store._collection_name)



    vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Keyword search retriever using BM25
    keyword_retriever = BM25Retriever.from_texts(vector_store.get()['documents'])
    embedding_retriever = EnsembleRetriever(retrievers=[keyword_retriever, vector_retriever], weights=[0.5, 0.5])
    # embedding_retriever = MultiQueryRetriever.from_llm(
    #     retriever=vector_store.as_retriever(), llm=llm
    # )
    # embedding_retriever = vector_store.as_retriever()

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=embedding_retriever
    )

    # embedding_retriever = vector_store.as_retriever(
    #                                                 )
    rag_chain = create_rag_chain(user_id, llm, compression_retriever)

    conversation_chain = create_conversation_chain(user_id, llm)
    return (vector_store, embedding_retriever, llm, rag_chain, conversation_chain)


def create_rag_chain(user_id, llm, embedding_retriever):
    # (1)
    # Создадим цепочку create_stuff_documents_chain, которая является частью
    # вопросно-ответной системы, ответственной за вставку фрагментов текстов
    # из векторной БД в промпт языковой модели
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=rag_prompt
    )
    # Создадим вопросно-ответную цепочку с помощью функции create_retrieval_chain().
    # используем ретривет для векторной базы с книгой

    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Учитывая приведённый ниже разговор, сгенерируйте поисковый запрос для поиска информации, относящейся к разговору. Ответьте только запросом, ничего больше."),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    # output_parser = StrOutputParser()

    # splitter = TokenTextSplitter(chunk_size=CHAT_DOC_SPLIT_SIZE, chunk_overlap=0)
    embeddings_filter = EmbeddingsFilter(
        embeddings = GigaChatEmbeddings(credentials=sber, verify_ssl_certs=False),
        similarity_threshold=0.1
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, embeddings_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=embedding_retriever
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | compression_retriever,
        ),
        query_transform_prompt | llm | output_parser | compression_retriever,
    ).with_config(run_name="chat_retriever_chain")

    # retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
    return (query_transforming_retriever_chain)


# Создать цепочку диалога
####################################################
def create_conversation_chain(user_id, llm):
    # (2)
    # Создать объект цепочки диалога - инициализация ConversationChain
    conversation = ConversationChain(llm=llm,
                                     verbose=True,
                                     # memory=ConversationSummaryBufferMemory(max_tokens=100))
                                     memory=ConversationBufferMemory())

    # Обращение к системе
    # conversation.predict(input='Как меня зовут и чем я занимаюсь?')
    return (conversation)


####################################################
# Загрузка и векторизация текста
def learn_document(doc_file, vector_store):
    # ЗАГРУЗКА И НАРЕЗКА ТЕКСТА DOCX
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                                   chunk_overlap=128)
    # Определить расшиение файла - docx, pdf или fb2
    file_ext = doc_file[doc_file.rfind(".") + 1:]
    # Использовать соответствующий загрузчик
    if file_ext == 'docx':
        loader = UnstructuredWordDocumentLoader(doc_file)
    elif file_ext == 'pdf':
        loader = UnstructuredPDFLoader(doc_file)
    elif file_ext == 'fb2':
        loader = UnstructuredXMLLoader(doc_file)
    else:
        return (0)

    splitted_data = loader.load_and_split(text_splitter)
    # ВЕКТОРИЗАЦИЯ
    # Добавить документ в векторное хранилище
    ids_list = vector_store.add_documents(splitted_data)

    return (len(ids_list))


####################################################################################################
#                                    Функции обработки команд                                      #
####################################################################################################

# `/start` - функция, обрабатывающая команду
#############################################
@bot.message_handler(commands=['start'])
def start(message: types.Message):
    user_id = message.chat.id

    # Проверка словарей для данного пользователя
    if user_id not in user_conversations:
        user_conversations[user_id] = ConversationBufferMemory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    bot.send_message(message.chat.id, 'Готов к работе')


# `/help` - функция, обрабатывающая команду
#############################################
@bot.message_handler(commands=['help'])
def help(message: types.Message):
    user_id = message.chat.id

    bot.send_message(message.chat.id,
                     'Я - бот-помощник в работе с текстами. Вы можете отправить мне файлы в форматах .doc .pdf .fb2, я их обработаю, загружу в векторную базу данных и Вы сможете задавать мне вопросы по этим текставм.')


####################################################################################################
#                                  Функции обработки сообщений                                     #
####################################################################################################

# Функция, обрабатывающая неправильные форматы ввода
####################################################
@bot.message_handler(content_types=['audio',
                                    'video',
                                    'photo',
                                    'sticker',
                                    'voice',
                                    'location',
                                    'contact'])
def not_text(message):
    user_id = message.chat.id
    bot.send_message(user_id, 'Я работаю только с текстовыми сообщениями и документами!')


# Функция, обрабатывающая файлы документов
##########################################
@bot.message_handler(content_types=['document'])
def handle_doc_message(message):
    user_id = message.chat.id

    # Проверка словарей для данного пользователя
    if user_id not in user_conversations:
        user_conversations[user_id] = ConversationBufferMemory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    # Загрузка файла
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_name = message.document.file_name
    bot.reply_to(message, "Читаю документ: " + file_name)
    # src = 'C:/Python/Project/tg_bot/files/received/' + file_name;
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)

    # Векторизация документа и добавление в векторное хранилище пользователя
    r = learn_document(file_name, vdb)
    # Сообщение пользователю о результате операции
    if r > 0:
        bot.send_message(user_id, "Документ прочитан (" + str(r) + ')')
        # Сохранить дополненное векторное хранилище
        vdb.save_local(doc_store, str(user_id))
    else:
        bot.send_message(user_id, "Не могу прочитать документ, вероятно не понятный формат.")

    sleep(2)


# Функция, обрабатывающая текстовые сообщения
#############################################
@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    user_id = message.chat.id

    # Проверка словарей для данного пользователя
    if user_id not in user_conversations:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # legacy_chain = LLMChain(
        #     llm=ChatOpenAI(),
        #     prompt=prompt,
        #     memory=memory,
        # )

        user_conversations[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]


    q1 = message.text
    print(q1)

    # (RAG)
    resp1 = rag_chain.invoke(
        {'input': q1}
    )

    # bot.send_message(user_id, 'RAG (' + q1 + '):')
    answer = resp1['answer']
    bot.send_message(user_id, answer)
    # (LLM)
    # q2 = answer
    # resp2 = conversation.predict(input=q2)
    # bot.send_message(user_id, 'LLM:')
    # bot.send_message(user_id, conversation.memory.chat_memory.messages[-1].content)

    # ........

    sleep(2)


bot.polling(none_stop=True)
