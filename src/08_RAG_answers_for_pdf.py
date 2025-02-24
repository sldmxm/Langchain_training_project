import csv
import hashlib
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini-2024-07-18'


def calculate_file_hash(file_path: str) -> Optional[str]:
    """Рассчитывает хеш-сумму файла для отслеживания изменений"""
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash


def process_document(
    pdf_path: str,
    persist_directory: str = './chroma_db',
    force_reprocess: bool = False,
) -> VectorStore:
    """Обработка PDF документа с сохранением эмбеддингов в векторную БД"""

    # Проверка необходимости пересчета
    hash_file = f'{persist_directory}/file_hash.txt'
    current_hash = calculate_file_hash(pdf_path)

    if (
        os.path.exists(persist_directory)
        and not force_reprocess
        and os.path.exists(hash_file)
    ):
        with open(hash_file, 'r') as f:
            saved_hash = f.read().strip()

        if saved_hash == current_hash:
            print('Документ уже обработан. Загружаем существующие эмбеддинги.')
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=OpenAIEmbeddings(),
            )

    print('Обрабатываем документ и создаем новые эмбеддинги...')

    # 1. Document loader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            '\n\n',
            '\n',
            '.',
            ' ',
        ],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Создание эмбеддингов и сохранение в векторную БД
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # Сохраняем хеш для последующих проверок
    os.makedirs(os.path.dirname(hash_file), exist_ok=True)
    if current_hash:
        with open(hash_file, 'w') as f:
            f.write(current_hash)

    return vectorstore


def answer_questions(
    questions_csv: str, output_csv: str, vectorstore: VectorStore
) -> None:
    """Отвечает на вопросы из CSV файла с помощью RAG и сохраняет результаты"""

    with open(questions_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        questions = [row[0] for row in reader]

    retriever = vectorstore.as_retriever(
        search_type='similarity', search_kwargs={'k': 3}
    )

    llm = ChatOpenAI(
        model_name=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0
    )

    template = """Ответь на вопрос, используя только предоставленный контекст.

    Контекст:
    {context}

    Вопрос: {question}

    Ответ:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: VectorStoreRetriever) -> str:
        return '\n\n'.join(
            f'Документ {i+1}:\n{doc.page_content}'
            for i, doc in enumerate(docs)
        )

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Отвечаем на вопросы и сохраняем результаты
    results = []

    for question in questions:
        answer = rag_chain.invoke(question)
        results.append([question, answer])

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Вопрос', 'Ответ'])
        writer.writerows(results)

    print(f'Результаты сохранены в {output_csv}')


def main(
    pdf_path: str = 'data/08_The_Daughter_of_The_Commandant.pdf',
    questions_csv: str = 'data/08_questions.csv',
    output_csv: str = 'data/08_results.csv',
    force_reprocess: bool = False,
) -> None:
    """Основная функция для запуска всего процесса"""

    # Обработка документа
    vectorstore = process_document(pdf_path, force_reprocess=force_reprocess)

    # Ответы на вопросы
    answer_questions(questions_csv, output_csv, vectorstore)
    # 14/15
    # Тупит с "Кого Пугачев оставил комендантом Белогорской крепости?"
    # "Пугачев оставил комендантом Белогорской крепости Ивана Кузьмича".


if __name__ == '__main__':
    main()
