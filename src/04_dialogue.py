import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini-2024-07-18'

llm = ChatOpenAI(
    temperature=0.0,
    api_key=OPENAI_API_KEY,
    model_name=MODEL_NAME,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            Always respond with just a single number,
            without any additional text or explanation.
            """,
        ),
        ('human', '{input}'),
    ]
)

store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


if __name__ == '__main__':
    df = pd.read_csv('data/04_dialogues.csv')
    df['question'] = df['question'].str.strip()

    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history,
    )

    answers = []
    wrong_counter = 0

    for dialogue_id, question, right_answer in tqdm(
        df[['dialogue_id', 'question', 'answer']].values
    ):
        response = chain.invoke(
            question, config={'configurable': {'session_id': str(dialogue_id)}}
        )
        answer = response.content
        answers.append(answer)

        if str(answer) != str(right_answer):
            wrong_counter += 1

    print(f'Total answers: {len(answers)}')
    print(f'Wrong answers: {wrong_counter}')
    print(f'Error rate: {(wrong_counter/len(answers))*100:.2f}%')
    # Total answers: 18
    # Wrong answers: 0
    # Error rate: 0.00%
