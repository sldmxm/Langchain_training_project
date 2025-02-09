import os

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

PROMPT_TEMPLATE = """
Input: "{input}"
Prompt: "Определи для объявления, сколько человек нуждается в жилье.
Если определить нельзя, ответь -1. В ответе должно быть одно целое число"
Output: ""
"""

# Можно брать батчами по 5-10 объявлений, сначала так и написал.
# Так быстрее и меньше токенов, но:
# (1) не всегда в ответе на 7 объявлений 7 ответов,
#   иногда выдает больше или меньше,
#   в итоге записать в файл датасета - отдельная проблема
# (2) ответы существенно точнее, если запрашивать по одному

if __name__ == '__main__':
    llm = ChatOpenAI(
        temperature=0.0,
        api_key=OPENAI_API_KEY,
        model_name='gpt-4o-mini-2024-07-18',
    )
    df = pd.read_csv('data/01_rent.csv')
    df.head()
    results = []
    for input_texts in tqdm(df['text']):
        prompt = PROMPT_TEMPLATE.format(input=input_texts)
        amount = int(llm.invoke(prompt).content)
        results.append(amount)

    df['predicted_people'] = results
    df.to_csv('data/01_rent_predicted.csv', index=False)
