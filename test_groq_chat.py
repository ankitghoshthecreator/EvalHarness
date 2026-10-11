from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

for model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Respond with 'Hello Groq' in 3 words"}],
            max_tokens=20,
        )
        print(f"[{model}] SUCCESS: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"[{model}] ERROR: {e}")
