import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration 
DATA_DIR      = os.getenv("OLD_TICKETS_DIR", "data/old_tickets")
PERSIST_DIR   = os.getenv("CHROMA_DIR", "chroma_store")
COLLECTION    = os.getenv("CHROMA_COLLECTION", "tickets")
MODEL_NAME    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RESET         = os.getenv("RESET_INDEX", "false").lower() in {"1", "true", "yes"}

# CSV column names
COL_ID         = os.getenv("COL_ID", "Ticket ID")
COL_TITLE      = os.getenv("COL_TITLE", "Issue")
COL_BODY       = os.getenv("COL_BODY", "Description")
COL_RESOLUTION = os.getenv("COL_RESOLUTION", "Resolution")
COL_CATEGORY   = os.getenv("COL_CATEGORY", "Category")
COL_DATE       = os.getenv("COL_DATE", "Date")
COL_AGENT      = os.getenv("COL_AGENT", "Agent Name")


def load_dataframe_from_file(path: Path) -> pd.DataFrame:
    """Load data from CSV or XLSX file into a DataFrame
    
    Args:
        path (Path): Path to the spreadsheet file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    ext = path.suffix.lower()
    logger.info(f"Loading spreadsheet file: {path}")
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported spreadsheet type: {path}")
    logger.info(f"Loaded {len(df)} rows from file: {path}")
    return df


def load_json_file(path: Path) -> List[Dict[str, Any]]:
    """Load JSON file where the structure is a dict of columns, each mapping row-IDs to values
    Returns a list of row-dicts with the same column keys

    Args:
        path (Path): Path to the JSON file.

    Returns:
        List[Dict[str, Any]]: List of row dictionaries.
    """
    logger.info(f"Loading JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must be a dict of columns: {path}")
    
    row_ids = set()
    for col, coldict in data.items():
        if isinstance(coldict, dict):
            row_ids.update(coldict.keys())

    rows: List[Dict[str, Any]] = []
    for row_id in sorted(row_ids):
        row_dict: Dict[str, Any] = {}
        for col_name, coldict in data.items():
            row_dict[col_name] = coldict.get(row_id, "")
        rows.append(row_dict)

    logger.info(f"Transformed JSON file to {len(rows)} row objects from: {path}")
    return rows


def row_to_doc(r: Union[pd.Series, Dict[str, Any]]) -> Document:
    """Convert a row into a Document
    
    Args:
        r (Union[pd.Series, Dict[str, Any]]): Row data

    Returns:
        Document: Converted document
    """
    text = (
        f"Title: {r.get(COL_TITLE, '')}\n\n"
        f"Description:\n{r.get(COL_BODY, '')}\n\n"
        f"Resolution:\n{r.get(COL_RESOLUTION, '')}"
    )
    metadata = {
        "id":       r.get(COL_ID, ""),
        "title":    r.get(COL_TITLE, ""),
        "category": r.get(COL_CATEGORY, ""),
        "date":     r.get(COL_DATE, ""),
        "agent":    r.get(COL_AGENT, ""),
    }

    return Document(page_content=text, metadata=metadata)


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Remove duplicates based on metadata 'id', preserving first occurrence
    
    Args:
        docs (List[Document]): List of documents to deduplicate.

    Returns:
        List[Document]: Deduplicated list of documents.
    """
    seen = set()
    unique: List[Document] = []
    for d in docs:
        doc_id = d.metadata.get("id")
        if doc_id in seen:
            logger.debug(f"Skipping duplicate doc id: {doc_id}")
            continue
        seen.add(doc_id)
        unique.append(d)
    logger.info(f"After deduplication: {len(unique)} unique documents")
    return unique


def main() -> None:
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    file_paths = list(data_path.glob("*"))
    logger.info(f"Processing {len(file_paths)} files in directory: {DATA_DIR}")

    docs: List[Document] = []
    for p in file_paths:
        try:
            if p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                df = load_dataframe_from_file(p)
                for _, row in df.iterrows():
                    docs.append(row_to_doc(row))
            elif p.suffix.lower() == ".json":
                entries = load_json_file(p)
                for r in entries:
                    docs.append(row_to_doc(r))
            else:
                logger.warning(f"Skipping unsupported file type: {p}")
        except Exception as e:
            logger.error(f"Failed processing file {p}: {e}", exc_info=True)

    logger.info(f"Loaded {len(docs)} raw documents from {DATA_DIR}")

    # Embeddings setup
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    if RESET and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)
        logger.info(f"Cleared existing Chroma directory: {PERSIST_DIR}")

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    docs = deduplicate_documents(docs)

    logger.info(f"Indexing {len(docs)} documents into collection '{COLLECTION}' at '{PERSIST_DIR}'")
    vectorstore.add_documents(docs)
    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()