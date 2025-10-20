# build_index.py
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# -------- Config (env or defaults) ----------
CSV_PATH = os.getenv("OLD_TICKETS_CSV", "data/old_tickets/ticket_dump_1.csv")
PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_store")
COLLECTION = os.getenv("CHROMA_COLLECTION", "tickets")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RESET = os.getenv("RESET_INDEX", "false").lower() in {"1", "true", "yes"}

# Column names in your CSV (change here if your headers differ)
COL_ID = os.getenv("COL_ID", "Ticket ID")
COL_TITLE = os.getenv("COL_TITLE", "Issue")
COL_BODY = os.getenv("COL_BODY", "Description")
COL_RESOLUTION = os.getenv("COL_RESOLUTION", "Resolution")
COL_CATEGORY = os.getenv("COL_CATEGORY", "Category")
COL_DATE = os.getenv("COL_DATE", "Date")
COL_AGENT = os.getenv("COL_AGENT", "Agent Name")


def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    return df


def row_to_doc(r: pd.Series) -> Document:
    # Compose the text field used for embedding
    text = (
        f"Title: {r.get(COL_TITLE, '')}\n\n"
        f"Description:\n{r.get(COL_BODY, '')}\n\n"
        f"Resolution:\n{r.get(COL_RESOLUTION, '')}"
    )
    metadata = {
        "id": r.get(COL_ID, ""),
        "title": r.get(COL_TITLE, ""),
        "category": r.get(COL_CATEGORY, ""),
        "date": r.get(COL_DATE, ""),
        "agent": r.get(COL_AGENT, ""),
    }
    return Document(page_content=text, metadata=metadata)


def main():
    df = load_df(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    if RESET and Path(PERSIST_DIR).exists():
        import shutil
        shutil.rmtree(PERSIST_DIR)
        print(f"Cleared existing Chroma directory: {PERSIST_DIR}")

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    docs = [row_to_doc(r) for _, r in df.iterrows()]
    seen = set()
    unique_docs = []
    for d in docs:
        doc_id = d.metadata.get("id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique_docs.append(d)
    docs = unique_docs
    print(f"Prepared {len(docs)} documents, adding to Chroma at '{PERSIST_DIR}'")
    vectorstore.add_documents(docs)
    print(f"Done. Collection='{COLLECTION}' at '{PERSIST_DIR}'")


if __name__ == "__main__":
    main()
