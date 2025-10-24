import os
import logging
import asyncio
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

import chainlit as cl
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain.agents.middleware import dynamic_prompt, ModelRequest, SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


# Configuration loader
def load_config() -> Dict[str, Any]:
    load_dotenv()
    return {
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "CHROMA_DIR": os.getenv("PERSIST_DIR", "chroma_store"),
        "COLLECTION": os.getenv("COLLECTION", "tickets"),
        "CHAT_MODEL": os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "MAX_NEW_TOKENS": int(os.getenv("MAX_NEW_TOKENS", "800")),
        "TEMPERATURE": float(os.getenv("TEMPERATURE", "0.2")),
    }


# Service builders
def build_retriever(cfg: Dict[str, Any]):
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg["EMBED_MODEL"], 
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        collection_name=cfg["COLLECTION"],
        embedding_function=embeddings,
        persist_directory=cfg["CHROMA_DIR"],
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def build_chat_model(cfg: Dict[str, Any]) -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
        repo_id=cfg["CHAT_MODEL"],
        task="text-generation",
        max_new_tokens=cfg["MAX_NEW_TOKENS"],
        do_sample=True,
        temperature=cfg["TEMPERATURE"],
        huggingfacehub_api_token=cfg["HF_TOKEN"],
        provider="auto",
    )
    return ChatHuggingFace(llm=llm, verbose=True)


def build_rag_agent(cfg: Dict[str, Any]):
    retriever = build_retriever(cfg)
    chat_model = build_chat_model(cfg)

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        user_query = request.state["messages"][-1].text
        docs = retriever.invoke(user_query)

        # Filter to unique IDs
        seen_ids = set()
        unique_docs = []
        for doc in docs:
            ticket_id = doc.metadata.get("id") or "unknown"
            if ticket_id in seen_ids:
                continue
            seen_ids.add(ticket_id)
            unique_docs.append((ticket_id, doc.page_content))

        # Format context
        context = "\n\n".join(
            f"- Ticket: {tid}\n{content}"
            for (tid, content) in unique_docs
        )

        return f"""
            You are an AI assistant working at Aleph Alpha, helping IT service agents handle incoming help-desk tickets.

            Your task:
            1. Read the incoming ticket description.
            2. Read the summaries of relevant past tickets (each with its unique Ticket ID).
            3. **Always reference** the Ticket IDs from the past tickets when you mention them.
            4. Provide a **concise, professional suggestion** (not a full solution) that helps the agent decide on next steps.

            Incoming ticket:
            {user_query}

            Relevant past tickets:
            {context}

            Instructions:
            - Use information from the relevant past tickets to guide your suggestion.
            - Do **not** invent a complete answer, you’re providing **direction** only.
            - Maintain a professional, clear tone appropriate for an IT service desk environment.
            - Limit your answer to about 3-5 sentences.
            - Always end your response with a list of the Ticket IDs you referenced, in the format: "Referenced Ticket IDs: [ID1, ID2, ...]"
            - At the end of your response, ask the user if they need further assistance.
        """

    rag_agent = create_agent(
        chat_model,
        tools=[],
        middleware=[
            prompt_with_context,
            SummarizationMiddleware(
                model=chat_model,
                max_tokens_before_summary=4000,
                messages_to_keep=20,
            ),
        ],
        checkpointer=InMemorySaver(),
    )

    return rag_agent


@cl.on_chat_start
async def start_chat():
    await cl.Message(content="Welcome! Describe your IT helpdesk ticket below to get assistance").send()


@cl.on_message
async def handle_message(message: cl.Message):
    user_query = message.content
    rag_agent = build_rag_agent(load_config())

    response = await rag_agent.ainvoke(
        {"messages": [{"role": "user", "content": user_query}]},
        {"configurable": {"thread_id": "helpdesk-session"}}
    )
    # Assume response["messages"][-1].content is the assistant reply
    await cl.Message(content=response["messages"][-1].content).send()


def main():
    logger = setup_logging()
    logger.info("Starting Chainlit RAG IT-Helpdesk App...")
    cfg = load_config()
    build_rag_agent(cfg)
    logger.info("RAG agent ready")
