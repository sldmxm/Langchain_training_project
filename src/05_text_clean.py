import os
import re

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
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


def del_chars(inputs: dict[str, str]) -> dict[str, str]:
    text = re.sub(r'[¿¡£]', '', inputs['text'])
    return {'text': text}


def check_answers(answers: list[dict[str, str]]) -> None:
    df_answers = pd.read_csv('data/05_answers.csv')
    wrong_counter = 0
    for answer, true_answer in zip(answers, df_answers.values, strict=False):
        if (
            answer['language'] != true_answer[1]
            or answer['hero'] != true_answer[2]
        ):
            print(f'{answer=}\n{true_answer[1:]=}')
            wrong_counter += 1

    print(f'Total texts processed: {len(answers)}')
    print(f'Wrong answers: {wrong_counter}')
    print(f'Error rate: {(wrong_counter/len(answers))*100:.2f}%')
    # Total texts processed: 13
    # Wrong answers: 0
    # Error rate: 0.00%


if __name__ == '__main__':
    lang_schema = ResponseSchema(
        name='language',
        description="""
            What language is the text written in?
            Answer one complete word in English.
            """,
    )
    hero_schema = ResponseSchema(
        name='hero',
        description="""
            Write the name of the main character.
            Answer in one word.
            """,
    )
    response_schema = [lang_schema, hero_schema]
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schema
    )

    prompt = PromptTemplate(
        template="""
            Text:
            {clear_text}

            Answer the questions:
            {format_instructions}
            """,
        input_variables=['input_text'],
        partial_variables={
            'format_instructions': output_parser.get_format_instructions()
        },
    )

    chain = {'clear_text': del_chars} | prompt | llm | output_parser

    with open('data/05_raw_texts.csv') as f:
        texts = [line.strip() for line in f.readlines()[1:]]

    answers = []
    for text in tqdm(texts):
        answers.append(chain.invoke({'text': text}))

    check_answers(answers)
