import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

import chainlit as cl
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain.agents.middleware import dynamic_prompt, ModelRequest, SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(__name__)

logger = setup_logging()


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
    logger.info(f"Building retriever with embed model {cfg['EMBED_MODEL']} and persist dir {cfg['CHROMA_DIR']}")
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg["EMBED_MODEL"], 
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        collection_name=cfg["COLLECTION"],
        embedding_function=embeddings,
        persist_directory=cfg["CHROMA_DIR"],
    )
    return vectorstore


def build_chat_model(cfg: Dict[str, Any]) -> ChatHuggingFace:
    logger.info(f"Building chat model {cfg['CHAT_MODEL']}")
    llm = HuggingFaceEndpoint(
        repo_id=cfg["CHAT_MODEL"],
        task="text-generation",
        max_new_tokens=cfg["MAX_NEW_TOKENS"],
        do_sample=True,
        temperature=cfg["TEMPERATURE"],
        huggingfacehub_api_token=cfg["HF_TOKEN"],
        provider="auto",
    )
    logger.info("Chat model ready")
    return ChatHuggingFace(llm=llm, verbose=True)


def build_rag_agent(cfg: Dict[str, Any]):
    logger.info("Building RAG agent...")
    retriever = build_retriever(cfg)
    chat_model = build_chat_model(cfg)

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        user_query = request.state["messages"][-1].text
        logger.info("Received user query: %s", user_query)

        docs_with_scores = retriever.similarity_search_with_score(user_query, k=3)
        logger.info(f"Retrieved {len(docs_with_scores)} past tickets for query")

        # Distance threshold
        MAX_DISTANCE = 1.105

        # Filter and ensure unique ticket IDs
        seen_ids = set()
        filtered_unique = []
        for doc, score in docs_with_scores:
            ticket_id = doc.metadata.get("id") or "unknown"
            if score > MAX_DISTANCE:
                logger.info(f"Skipping ticket {ticket_id} with distance {score} above threshold {MAX_DISTANCE}")
                continue
            if ticket_id in seen_ids:
                logger.debug(f"Skipping duplicate ticket id: {ticket_id}")
                continue
            seen_ids.add(ticket_id)
            filtered_unique.append((ticket_id, doc.page_content, score))

        # Fallback if no docs found
        if not filtered_unique:
            logger.info("No relevant past tickets found, using fallback prompt.")
            return """
                You are an AI assistant helping IT service agents with help-desk tickets.

                I could not find any previously resolved tickets closely matching this new incoming ticket.

                Incoming ticket:
                {user_query}

                Since no past similar cases were retrieved, acknowledge the lack of prior context. Then suggest that the agent:
                - ask the requester for additional details (for example: error logs, steps already tried, user environment)
                - consider escalating the ticket or consulting a subject-matter expert
                - reference that you had no prior similar cases and end with asking if the agent needs further assistance.

                Example ending: “Referenced Ticket IDs: []”
            """.format(user_query=user_query)

        logger.info(f"Found {len(filtered_unique)} past tickets for context")

        # Format context
        context = "\n\n".join(
            f"- Ticket: {tid} (distance: {score:.2f})\n{content}"
            for (tid, content, score) in filtered_unique
        )

        # Normal case 
        return f"""
            You are an AI assistant, helping IT service agents handle incoming help-desk tickets.

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
            - Use information from the relevant past tickets to guide your suggestion
            - Do **not** invent a complete answer, you’re providing **direction** only
            - Maintain a professional, clear tone appropriate for an IT service desk environment
            - If you are asked about something not related to IT helpdesk tickets, respond with "I'm sorry, I can only assist with IT helpdesk tickets."
            - Always end your response with a list of the Ticket IDs you referenced, in the format: "Referenced Ticket IDs: [ID1, ID2, ...]"
            - At the end of your response, ask the user if they need further assistance
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
    logger.info("RAG agent built successfully")

    return rag_agent


@cl.on_chat_start
async def start_chat():
    logger.info("Starting Chainlit chat session...")
    rag_agent = build_rag_agent(load_config())
    cl.user_session.set("agent", rag_agent)
    logger.info("RAG agent stored in user session")


@cl.on_stop
def on_stop():
    logger.info("Chat session stopped")


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="VPN keeps disconnecting",
            message="A user reports the VPN connection drops after 5-10 minutes of use. What direction can you give the agent based on past ticket history?",
            icon="/public/network.svg",
        ),
        cl.Starter(
            label="Shared drive access denied",
            message="An employee cannot access the shared company drive despite having permissions. Suggest a next step for the service agent.",
            icon="/public/drive.svg",
        ),
        cl.Starter(
            label="Laptop battery not charging",
            message="A user’s laptop battery is not charging even when plugged in. Provide guidance for the agent on how to move forward.",
            icon="/public/battery.svg",
        ),
    ]



@cl.on_message
async def handle_message(message: cl.Message):
    user_query = message.content
    logger.info(f"Received user message: {user_query}")
    rag_agent = cl.user_session.get("agent")

    response = await rag_agent.ainvoke(
        {"messages": [{"role": "user", "content": user_query}]},
        {"configurable": {"thread_id": "helpdesk-session"}}
    )
    logger.info("Sending response back to user")
    await cl.Message(content=response["messages"][-1].content).send()


def main():
    logger = setup_logging()
    logger.info("Starting Chainlit RAG IT-Helpdesk App...")

    from chainlit.cli import run_chainlit
    run_chainlit(__file__)


if __name__ == "__main__":
    main()
