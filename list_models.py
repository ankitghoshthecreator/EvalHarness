from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
print(f"Key loaded: {api_key[:8]}...{api_key[-4:] if api_key else 'None'}")

try:
    from groq import Groq
    client = Groq(api_key=api_key)
    models = client.models.list()
    print(f"Found {len(models.data)} models available to this API key:")
    for m in sorted(models.data, key=lambda x: x.id):
        print(f"  - {m.id}")
except Exception as e:
    print(f"Error querying Groq: {e}")
