import json
import requests
from retrieve import retrieve

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "qwen2.5:1.5b"

def compress_context(chunks, max_chars=700):
    blocks = []

    for c in chunks:
        blocks.append(
            f"[Source: {c['source']}]\n{c['text']}"
        )

    combined = "\n\n".join(blocks)
    return combined[:max_chars]

def ask(query):
    context = retrieve(query)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context. If no context, say: "No context given".
Cite the source document name in your answer.
If the answer is not in the context, say: "I do not know".

Context:
{compress_context(context)}

Question:
{query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 128,
                "repeat_penalty": 1.1,
            }
        },
        stream=True,
        timeout=None
    )

    answer = ""
    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))

        if "response" in data:
            print(data["response"], end="", flush=True)
            answer += data["response"]

    print()
    return answer

if __name__ == "__main__":
    while True:
        q = input(">> ")
        answer = ask(q)

