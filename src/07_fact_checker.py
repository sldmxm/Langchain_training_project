import csv
import os
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.human.tool import HumanInputRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini-2024-07-18'

Answer = Literal['true', 'false', 'uncertain']


@tool
def answer_question(question: str) -> Answer:
    """Отвечает на вопрос, используя базу знаний модели.
    Возвращает 'true', 'false' или 'uncertain'"""
    llm = ChatOpenAI(
        temperature=0.0, api_key=OPENAI_API_KEY, model_name=MODEL_NAME
    )
    response = llm.invoke(
        f"""Ответь 'true', 'false' или 'uncertain'
        на следующее утверждение: {question}"""
    )
    result = response.content.strip().lower()
    return cast(Answer, result if result in ('true', 'false') else 'uncertain')


class FactChecker:
    def __init__(self, model_name: str = MODEL_NAME, verbose: bool = True):
        self.llm = ChatOpenAI(
            temperature=0.0,
            api_key=OPENAI_API_KEY,
            model_name=model_name,
        )
        self.verbose = verbose
        self.agent_executor = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """Ты - агент для проверки фактов.
            Твоя задача - определить, является ли утверждение
            истинным или ложным.

            Следуй этому алгоритму:
            1. Если это утверждение общего характера,
            используй answer_question
            2. Если answer_question вернул 'uncertain',
            используй human для запроса у человека

            В итоге ты должен вернуть только 'True' или 'False'.""",
                ),
                MessagesPlaceholder(variable_name='chat_history'),
                ('human', '{input}'),
                MessagesPlaceholder(variable_name='agent_scratchpad'),
            ]
        )

        tools = [
            answer_question,
            HumanInputRun(),
        ]

        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            max_iterations=3,
            handle_parsing_errors=True,
        )

    def check_statement(self, statement: str) -> bool:
        """Проверяет одно утверждение и возвращает результат"""
        response = self.agent_executor.invoke(
            {'input': statement, 'chat_history': []}
        )

        try:
            answer = str(response['output'])
            return answer.lower() == 'true'
        except Exception as e:
            if self.verbose:
                print(f'Ошибка при обработке ответа: {e}')
            return False

    def check_from_csv(self, csv_path: str) -> list[dict[str, Any]]:
        """Проверяет все утверждения из CSV файла"""
        results = []

        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                statement = row['texts']
                expected = row['answers'].lower() == 'true'

                answer = self.check_statement(statement)
                results.append(
                    {
                        'statement': statement,
                        'agent_answer': answer,
                        'expected_answer': expected,
                        'is_correct': answer == expected,
                    }
                )

        return results


def print_results(results: list[dict[str, Any]]) -> None:
    """Выводит результаты проверки в читаемом формате"""
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)

    print(f'\nПроверено правильно: {correct_count} из {total_count}')
    print(f'Точность: {(correct_count/total_count)*100:.2f}%\n')
    # Проверено правильно: 25 из 25
    # Точность: 100.00%

    for i, result in enumerate(results, 1):
        print(f'Утверждение {i}:')
        print(f"Текст: {result['statement']}")
        print(f"Ответ агента: {result['agent_answer']}")
        print(f"Правильный ответ: {result['expected_answer']}")
        print(f"Статус: {'✓ Верно' if result['is_correct'] else '✗ Неверно'}")
        print('-' * 50)


def main() -> None:
    csv_path = 'data/07_fact_checker.csv'

    checker = FactChecker(verbose=True)
    results = checker.check_from_csv(csv_path)
    print_results(results)


if __name__ == '__main__':
    main()
