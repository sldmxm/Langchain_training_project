import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    api_key=openai_api_key,
)
messages = [
    {
        'role': 'user',
        'content': '1+1',
    }
]
chat_completion = client.chat.completions.create(
    max_tokens=100,
    model='gpt-4o-mini-2024-07-18',
    messages=messages,
    temperature=0,
)
print(chat_completion.choices[0].message.content)
