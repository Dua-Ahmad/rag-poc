import json
import requests
from retrieve import retrieve, list_documents, documents_mentioning

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

def extract_topic(query):
    q = query.lower()

    for phrase in ["talk about", "mention", "discuss", "about"]:
        if phrase in q:
            return q.split(phrase)[-1].strip()

    return query


def ask(query):

    q = query.lower()

    # ---------- GENERIC document listing ----------
    if any(w in q for w in ["list", "give me", "show me"]) and \
       any(w in q for w in ["document", "documents", "files", "presentations"]):

        docs = list_documents()

        if not docs:
            return "No documents found in the system."

        lines = ["Documents in the system:"]
        for name, ext in sorted(docs.items()):
            lines.append(f"- {name} ({ext})")

        return "\n".join(lines)
    
    # ---------- DOCUMENTS THAT MENTION X ----------
    if "document" in q and any(w in q for w in ["talk about", "mention", "discuss"]):
        topic = extract_topic(query)
        grouped = documents_mentioning(topic)

        if not grouped:
            return f"No documents mention '{query}'."

        lines = [f"Documents that mention '{topic}':"]

        for doc, chunks in grouped.items():
            lines.append(f"\n- {doc}")

            seen = set()

            for c in chunks:  # show evidence
                snippet = c["text"][:120].strip()
                
                if snippet in seen:
                      continue
                
                seen.add(snippet)
                lines.append(f"    • ...{snippet}...")  

                if len(seen) == 2:
                    break

        return "\n".join(lines)


    # ---------- Semantic RAG ----------

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
        if answer:
            print(answer)
