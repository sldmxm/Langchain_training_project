import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini-2024-07-18'


PROMPT_TEMPLATE = """
Сообщение из чата по криптовалютам: {message}.
Ответь "1", если это спам, иначе "0"
"""

llm = ChatOpenAI(
    temperature=0.0,
    api_key=OPENAI_API_KEY,
    model_name=MODEL_NAME,
)


def one_by_one_process_spam(df: pd.DataFrame) -> None:
    prompt_template = PromptTemplate(
        input_variables=['message'], template=PROMPT_TEMPLATE
    )

    wrong_counter = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for message, true_is_spam in tqdm(
        zip(df['text'], df['is_spam'], strict=False), total=len(df)
    ):
        prompt = prompt_template.format(message=message)
        total_input_tokens += llm.get_num_tokens(prompt)
        resp = llm.invoke(prompt).content
        total_output_tokens += llm.get_num_tokens(resp)
        wrong_counter += int(resp) != int(true_is_spam)

    error_rate = wrong_counter / len(df) * 100
    print(f'Доля неправильных ответов: {error_rate:.2f}%')
    print(f'Всего токенов (вход): {total_input_tokens}')
    print(f'Всего токенов (выход): {total_output_tokens}')
    # Доля неправильных ответов: 3.00%
    # Всего токенов (вход): 7147
    # Всего токенов (выход): 100
    # 01:16 полное время


def batch_process_spam(df: pd.DataFrame, batch_size: int = 10) -> None:
    prompt_template = PromptTemplate(
        input_variables=['message'], template=PROMPT_TEMPLATE
    )
    messages = [prompt_template.format(message=m) for m in df['text']]
    responses = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i in tqdm(
        range(0, len(messages), batch_size),
        total=(len(messages) // batch_size)
        + (1 if len(messages) % batch_size else 0),
    ):
        batch = messages[i : i + batch_size]

        batch_input_tokens = sum(llm.get_num_tokens(msg) for msg in batch)
        total_input_tokens += batch_input_tokens

        batch_responses = llm.generate(batch)
        batch_outputs = [
            r[0].text.strip() for r in batch_responses.generations
        ]
        responses.extend(batch_outputs)

        batch_output_tokens = sum(
            llm.get_num_tokens(resp) for resp in batch_outputs
        )
        total_output_tokens += batch_output_tokens

    df['predicted'] = [int(resp.strip()) for resp in responses]

    for message, true_is_spam, is_spam in zip(
        df['text'], df['is_spam'], df['predicted'], strict=False
    ):
        if true_is_spam != is_spam:
            print(
                f'Сообщение: {message} '
                f'Правильный ответ: {true_is_spam} '
                f'Мой ответ: {is_spam}'
            )

    wrong_counter = (df['predicted'] != df['is_spam']).sum()
    error_rate = wrong_counter / len(df) * 100

    print(f'Доля неправильных ответов: {error_rate:.2f}%')
    print(f'Всего токенов (вход): {total_input_tokens}')
    print(f'Всего токенов (выход): {total_output_tokens}')
    # Доля неправильных ответов: 2.00%
    # Всего токенов (вход): 7147
    # Всего токенов (выход): 100
    # 01:07 полное время


if __name__ == '__main__':
    df = pd.read_csv('data/02_crypto_spam.csv')
    df.head()
    df['text'] = df['text'].str.strip()
    one_by_one_process_spam(df)
    batch_process_spam(df)
