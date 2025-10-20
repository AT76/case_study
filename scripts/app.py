import os
import logging
import asyncio
from typing import AsyncGenerator, List, Optional, Any

from dotenv import load_dotenv

import chainlit as cl
from huggingface_hub import AsyncInferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Config
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("PERSIST_DIR", "chroma_store")
COLLECTION = os.getenv("COLLECTION", "tickets")

CHAT_MODEL = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_PROVIDER = os.getenv("HF_PROVIDER", "novita")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


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
# Create client lazily so the app can start even if HF creds are missing.
client: Optional[AsyncInferenceClient] = None
if HF_PROVIDER:
    try:
        client = AsyncInferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
    except Exception as e:
        logger.warning("Failed to initialize HF AsyncInferenceClient: %s", e)
        client = None


async def generate_answer_via_api(question: str, context: str) -> AsyncGenerator[str, None]:
    if client is None:
        raise RuntimeError("HF AsyncInferenceClient not configured. Set HF_TOKEN/HF_PROVIDER.")

    prompt_context = (
        f"Here are similar solved tickets:\n{context}\n\n"
        f"New ticket: {question}\n\n"
        "Please suggest a helpful direction (steps or checks) the agent should take. "
        "Don't answer questions that are not related to IT helpdesk support."
    )
    messages = [
        {"role": "system", "content": "You are an IT helpdesk assistant."},
        {"role": "user", "content": prompt_context},
    ]

    resp_stream = None
    closed = False
    try:
        resp_stream = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )

        def _extract_content_from_chunk(chunk: Any) -> Optional[str]:
            if not chunk or not getattr(chunk, "choices", None):
                return None
            delta = getattr(chunk.choices[0], "delta", None)
            return getattr(delta, "content", None) if delta else None

        async for chunk in resp_stream:
            if closed:
                break
            content = _extract_content_from_chunk(chunk)
            if content:
                yield content

    except (asyncio.CancelledError, GeneratorExit):
        closed = True
        if resp_stream is not None:
            try:
                asyncio.create_task(resp_stream.aclose())
            except Exception:
                pass
        return
    except Exception as exc:
        logger.error("Error during API call or streaming: %s", exc)
        yield f"\n\n[Error: {exc}]\n"
        return
    finally:
        # No yields here
        if resp_stream is not None:
            try:
                asyncio.create_task(resp_stream.aclose())
            except Exception:
                logger.warning("Failed to schedule response stream close cleanly.")


def _retrieve_similar_documents(query: str) -> List[Document]:
    """
    Retrieve similar documents using the retriever. Handles common retriever APIs:
    - get_relevant_documents
    - retrieve
    - invoke

    Returns:
        list of Document
    Raises:
        RuntimeError if no supported retrieval method is found.
    """
    # Try common LangChain retriever methods (they are usually synchronous)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query)
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    raise RuntimeError("Retriever does not implement a supported interface.")


@cl.on_chat_start
async def start():
    """Send an initial greeting when the chat starts."""
    await cl.Message(
        content=(
            "Hello! Paste your IT-ticket (symptoms + environment + what's been tried) "
            "and I'll suggest a direction."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Main handler for incoming user messages.

    Steps:
    1) Retrieve similar tickets from the vector store.
    2) De-duplicate by ticket id.
    3) Stream model completion tokens back to the client.
    """
    user_q = (message.content or "").strip()
    if not user_q:
        await cl.Message(content="Please provide the ticket details.").send()
        return

    # 1) Retrieve similar tickets (defensive wrapper to support multiple retriever implementations)
    try:
        docs = _retrieve_similar_documents(user_q) or []
    except Exception as e:
        logger.exception("Failed to retrieve documents: %s", e)
        await cl.Message(content=f"Error retrieving similar tickets: {e}").send()
        return

    # 2) Filter duplicates by ticket id (if metadata contains "id")
    filtered: List[Document] = []
    seen = set()
    for d in docs:
        tid = None
        try:
            tid = d.metadata.get("id") if getattr(d, "metadata", None) else None
        except Exception:
            tid = None
        if tid and tid in seen:
            continue
        if tid:
            seen.add(tid)
        filtered.append(d)
    docs = filtered

    # Build context string to feed to the generation prompt
    context_str = "\n\n---\n\n".join(getattr(d, "page_content", "") for d in docs)

    # 3) Generate answer via streaming
    try:
        msg = await cl.Message(content="").send()
        async for token in generate_answer_via_api(user_q, context_str):
            try:
                await msg.stream_token(token)
            except asyncio.CancelledError:
                break   # user closed UI; stop cleanly
            except Exception:
                logger.exception("Failed to stream token; continuing.")
        await msg.update()
    except Exception as e:
        logger.exception("Error during generation/streaming: %s", e)
        await cl.Message(content=f"An error occurred while generating a response: {e}").send()
