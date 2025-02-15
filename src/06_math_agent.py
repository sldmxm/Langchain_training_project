import csv
import math
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini-2024-07-18'


@tool
def add_numbers(num1: float, num2: float) -> float:
    """Складывает два числа"""
    return num1 + num2


@tool
def convert_meters_to_cm(num: float) -> float:
    """Переводит метры в сантиметры"""
    return num * 100


@tool
def convert_cubic_cm_to_liters(num: float) -> float:
    """Переводит кубические сантиметры в литры"""
    return num / 1000


@tool
def compute_rectangle_perimeter(a: float, b: float) -> float:
    """Вычисляет периметр прямоугольника"""
    return 2 * (a + b)


@tool
def compute_circle_area(radius: float) -> float:
    """Вычисляет площадь круга"""
    return math.pi * radius**2


@tool
def compute_cylinder_volume(radius: float, height: float) -> float:
    """Вычисляет объем цилиндра"""
    return math.pi * radius**2 * height


@tool
def compute_cube_volume(a: float) -> float:
    """Вычисляет объем куба"""
    return a**3


@tool
def convert_binary_to_decimal(binary_number: str) -> int:
    """Переводит число из двоичной системы счисления в десятичную"""
    return int(binary_number, 2)


@tool
def convert_decimal_to_binary(decimal_number: int) -> str:
    """Переводит число из десятичной системы счисления в двоичную"""
    return bin(decimal_number)[2:]


@tool
def get_count_ones(number: str) -> int:
    """Находит количество единиц в двоичном представлении числа.
    На вход получает строку - двоичное число"""
    return str(number).count('1')


def create_math_agent() -> AgentExecutor:
    llm = ChatOpenAI(
        temperature=0.0,
        api_key=OPENAI_API_KEY,
        model_name=MODEL_NAME,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """Ты - математический помощник.
         Твоя задача - решать математические задачи,
         используя доступные инструменты.
         Внимательно изучи задание, построй план,
         выполни преобразования, если они необходимы.
         В качестве ответа верни ТОЛЬКО одно число без каких-либо пояснений
         и дополнительного текста. Число должно быть в формате float.
         """,
            ),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}'),
            MessagesPlaceholder(variable_name='agent_scratchpad'),
        ]
    )

    tools = [
        add_numbers,
        convert_meters_to_cm,
        convert_cubic_cm_to_liters,
        compute_rectangle_perimeter,
        compute_circle_area,
        compute_cylinder_volume,
        compute_cube_volume,
        convert_binary_to_decimal,
        convert_decimal_to_binary,
        get_count_ones,
    ]

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=7,
        handle_parsing_errors=True,
    )
    return agent_executor


def solve_math_problems(csv_path: str) -> list[dict[str, float | str | bool]]:
    agent = create_math_agent()
    results = []

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            problem = row['problem']
            expected_answer = float(row['answer'])

            response = agent.invoke({'input': problem, 'chat_history': []})
            try:
                answer = float(response['output'])
                is_correct = abs(answer - expected_answer) < 1e-6
            except ValueError:
                is_correct = False

            results.append(
                {
                    'problem': problem,
                    'agent_answer': answer,
                    'expected_answer': expected_answer,
                    'is_correct': is_correct,
                }
            )

    return results


def main() -> None:
    csv_path = 'data/06_math_problems.csv'
    results = solve_math_problems(csv_path)

    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)

    print(f'\nРешено правильно: {correct_count} из {total_count}')
    print(f'Точность: {(correct_count/total_count)*100:.2f}%')
    # Решено правильно: 8 из 10
    # Точность: 80.00%

    for i, result in enumerate(results, 1):
        print(f'\nЗадача {i}:')
        print(f"Условие: {result['problem']}")
        print(f"Ответ агента: {result['agent_answer']}")
        print(f"Правильный ответ: {result['expected_answer']}")
        print(f"Статус: {'Верно' if result['is_correct'] else 'Неверно'}")


if __name__ == '__main__':
    main()
