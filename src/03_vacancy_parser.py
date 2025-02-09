import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
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


@dataclass
class Vacancy:
    job_title: str
    company: str
    salary: str
    tg: str
    grade: str


def parse_vacancy(message: str) -> Vacancy:
    prompt_template = PromptTemplate(
        input_variables=['message'],
        template="""
        Извлеки информацию из вакансии: {message}
        Если какой-то информации нет, укажи ""
        {format_instructions}
        """,
    )

    job_title_schema = ResponseSchema(
        name='job_title', description='Название должности без грейда'
    )
    company_schema = ResponseSchema(
        name='company',
        description="""
        Название компании. Тип, например, "финтех" или "банк" не указывать
        """,
    )
    salary_schema = ResponseSchema(
        name='salary',
        description="""
        Уровень оплаты.
        Числа без пробелов, не пишем тыс. или к. - приводим к единицам.
        Игнорируем неденежное вознаграждение.
        Не пишем net, gross, на руки.
        Валюту указываем через пробел после суммы.
        Для диапазона, то пишем его через тире,
        около тире пробелы не ставим
        (например, 100к-150к рублей -> 100000-150000 руб.).
        Если только нижняя граница (от 2000$ или 100000+руб),
        указывем "от" (например, 100к+руб -> от 100000 руб.).
        Если только верхняя граница,
        указываем "до" (например, до 100к руб -> до 100000 руб.).
        Если зарплата указана за час, то в конце добавляем "в час".
        """,
    )
    tg_schema = ResponseSchema(
        name='tg',
        description="""
        Контакт для связи.
        Указываем контакты, используя @, списком'
        """,
    )
    grade_schema = ResponseSchema(
        name='grade',
        description="""
        Грейд.
        Возможные значения:
        "", intern, junior, junior+, middle, middle+, senior, lead.
        Если несколько, пишем списком в порядке возрастания.
        Если руководящая должность, указываем "lead".
        Если не указан явно и не уверены, указываем ""
        """,
    )
    response_schemas = [
        job_title_schema,
        company_schema,
        salary_schema,
        tg_schema,
        grade_schema,
    ]
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas
    )
    format_instructions = output_parser.get_format_instructions()

    prompt = prompt_template.format(
        message=message, format_instructions=format_instructions
    )
    response = llm.invoke(prompt)

    return Vacancy(**output_parser.parse(response.content))


if __name__ == '__main__':
    PARAM_NUM = 5

    df = pd.read_csv('data/03_vacancies.csv')
    df.head()
    df['text'] = df['text'].str.strip()

    wrong_counter = 0
    for text, *correct in tqdm(
        df[['text', 'job_title', 'company', 'salary', 'tg', 'grade']].values
    ):
        correct = ['' if pd.isna(value) else value for value in correct]
        right_result = Vacancy(*correct)

        result = parse_vacancy(text)

        if right_result != result:
            differences = {
                field: (getattr(right_result, field), getattr(result, field))
                for field in right_result.__dict__
                if getattr(right_result, field) != getattr(result, field)
            }
            wrong_counter += len(differences)

            # print('Отличия:')
            # for field, (val1, val2) in differences.items():
            # print(f"{field}: {val1} → {val2}")

    error_rate = 100 * wrong_counter / (len(df) * PARAM_NUM)
    print(f'Неправильных ответов: {wrong_counter}')
    print(f'Доля неправильных ответов: {error_rate:.2f}%')
    # Неправильных ответов: 44
    # Доля неправильных ответов: 17,6%
