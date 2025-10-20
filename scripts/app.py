import os
from dotenv import load_dotenv

import chainlit as cl
from huggingface_hub import AsyncInferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Config 
EMBED_MODEL    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR     = os.getenv("PERSIST_DIR",    "chroma_store")
COLLECTION     = os.getenv("COLLECTION", "tickets")

CHAT_MODEL     = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN       = os.getenv("HF_TOKEN", None)
HF_PROVIDER    = os.getenv("HF_PROVIDER", "novita")  

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.2"))


# Retrieval Setup
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Generation Setup
client = AsyncInferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)


async def generate_answer_via_api(question: str, context: str) -> str:
    # Build system message + context
    prompt_context = (
        f"Here are similar solved tickets:\n{context}\n\n"
        f"New ticket: {question}\n\n"
        "Please suggest a helpful direction (steps or checks) the agent should take."
        "Dont answer questions that are not related to IT helpdesk support."
    )
    messages = [
        {"role": "system", "content": "You are an IT helpdesk assistant."},
        {"role": "user",   "content": prompt_context}
    ]
    resp_stream = await client.chat.completions.create(
        model       = CHAT_MODEL,
        messages    = messages,
        max_tokens  = MAX_NEW_TOKENS,
        temperature = TEMPERATURE,
        stream      = True
    )
    async for chunk in resp_stream:
        # safety check
        if not hasattr(chunk, "choices"):
            continue
        if len(chunk.choices) == 0:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        content = delta.get("content")
        if content:
            yield content


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! Paste your IT-ticket (symptoms + environment + whats been tried) and I'll suggest a direction"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_q = message.content.strip()

    # 1) Retrieve similar tickets
    docs = retriever.invoke(user_q)
    # Filter duplicates by ticket id
    filtered = []
    seen = set()
    for d in docs:
        tid = d.metadata.get("id")
        if tid in seen:
            continue
        seen.add(tid)
        filtered.append(d)
    docs = filtered
    context_str = "\n\n---\n\n".join(d.page_content for d in docs)

    # 3) Generate answer via streaming
    msg = await cl.Message(content="").send()  # send empty first so we can stream into it
    async for token in generate_answer_via_api(user_q, context_str):
        await msg.stream_token(token)
    await msg.update()  # finalise message
