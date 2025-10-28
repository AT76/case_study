import pytest
from pathlib import Path
from typing import List, Dict, Any

from src.build_index import (
    load_dataframe_from_file,
    load_json_file,
    row_to_doc,
    COLLECTION,
    MODEL_NAME,
    DATA_DIR
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from scripts.app import build_rag_agent, load_config  


def load_all_docs(data_dir: Path):
    docs = []
    for p in data_dir.glob("*"):
        if p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            df = load_dataframe_from_file(p)
            for _, r in df.iterrows():
                docs.append(row_to_doc(r))
        elif p.suffix.lower() == ".json":
            rows = load_json_file(p)
            for r in rows:
                docs.append(row_to_doc(r))
    return docs

@pytest.fixture(scope="module")
def vectorstore(tmp_path_factory):
    # build a temporary vectorstore from actual data for testing
    docs = load_all_docs(Path(DATA_DIR))
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    store_dir = tmp_path_factory.mktemp("chromatest")
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(store_dir),
    )
    vs.add_documents(docs)
    return vs

# Fixture for temp directory
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture(scope="module")
def rag_agent():
    cfg = load_config()
    return build_rag_agent(cfg)


NEW_TICKETS_CSV = Path("data/new_tickets.csv")

def load_new_tickets() -> List[Dict[str, Any]]:
    df_new = load_dataframe_from_file(NEW_TICKETS_CSV)
    inputs: List[Dict[str, Any]] = []
    for _, row in df_new.iterrows():
        inputs.append({
            "ticket_id": row["Ticket ID"],
            "issue": row["Issue"],
            "description": row["Description"],
            "category": row["Category"],
            "date": row["Date"],
            "ground_truth": [] 
        })
    return inputs

GROUND_TRUTH_MAP: Dict[str, List[str]] = {
    "TCKT-2000": ["TCKT-1011", "TKT1002", "TCKT-1051"],  # VPN
    "TCKT-2001": ["TCKT-1010", "TKT1000", "TCKT-1050"],  # Email syncing
    "TCKT-2002": ["TCKT-1013", "TKT1003", "TCKT-1052"],  # Software installation rights
    "TCKT-2003": ["TCKT-1045", "TKT1004", "TCKT-1055"],  # Laptop overheating (flickering)
    "TCKT-2004": ["TCKT-1047", "TKT1005", "TCKT-1057"],  # Shared drive access
    "TCKT-2005": ["TCKT-1044", "TKT1006", "TCKT-1054"],  # Printer connectivity
    "TCKT-2006": ["TCKT-1046", "TKT1007", "TCKT-1059"],  # Application crash/performance
    "TCKT-2007": ["TCKT-1012", "TKT1008", "TCKT-1076"],  # Lost password for multiple accounts
    "TCKT-2008": [],                                     # Server access
    "TCKT-2009": ["TCKT-1048", "TCKT-1078"],             # Spam filter blocking emails
}


def compute_hit_at_k(retrieved_ids: List[str], ground_truth: List[str], k: int) -> int:
    """Computes Hit@K: returns 1 if at least one ground-truth ID is found in the top K retrieved IDs, otherwise 0
    
    Args:
        retrieved_ids (List[str]): List of retrieved document IDs
        ground_truth (List[str]): List of ground-truth document IDs
        k (int): The cutoff rank K

    Returns:
        int: 1 if hit, 0 otherwise
    """
    return 1 if any(gt in retrieved_ids[:k] for gt in ground_truth) else 0


def compute_precision_at_k(retrieved_ids: List[str], ground_truth: List[str], k: int) -> float:
    """Compute Precision@K: fraction of the top K retrieved IDs that are relevant (in ground_truth)

    How many of the top K retrieved documents are relevant?
    
    Args:
        retrieved_ids (List[str]): List of retrieved document IDs
        ground_truth (List[str]): List of ground-truth document IDs
        k (int): The cutoff rank K

    Returns:
        float: Precision at K
    """
    if k == 0:
        return 0.0
    return len([rid for rid in retrieved_ids[:k] if rid in ground_truth]) / k


def compute_recall_at_k(retrieved_ids: List[str], ground_truth: List[str], k: int) -> float:
    """Computes Recall@K: fraction of the ground-truth IDs that appear in the top K retrieved IDs

    How many of the relevant documents were retrieved in the top K?
    
    Args:
        retrieved_ids (List[str]): List of retrieved document IDs
        ground_truth (List[str]): List of ground-truth document IDs
        k (int): The cutoff rank K

    Returns:
        float: Recall at K
    """
    if not ground_truth:
        return 0.0
    return len([rid for rid in retrieved_ids[:k] if rid in ground_truth]) / len(ground_truth)


def compute_mrr(retrieved_ids: List[str], ground_truth: List[str]) -> float:
    """Computes Mean Reciprocal Rank (MRR) for one query: the reciprocal of the rank of the first relevant document

    How soon is the first relevant document retrieved?
    
    Args:
        retrieved_ids (List[str]): List of retrieved document IDs
        ground_truth (List[str]): List of ground-truth document IDs
    
    Returns:
        float: MRR value
    """
    for idx, rid in enumerate(retrieved_ids, start=1):
        if rid in ground_truth:
            return 1.0 / idx
    return 0.0


def test_retrieval_metrics(vectorstore):
    """
    Test retrieval performance on new tickets using the vectorstore
    Computes metrics for each input and verifies expected behaviour
    """
    new_inputs = load_new_tickets()
    k = 3

    precision_scores = []
    recall_scores = []
    hit_scores = []
    mrr_scores = []

    for inp in new_inputs:
        gt = GROUND_TRUTH_MAP.get(inp["ticket_id"], [])
        retrieved = vectorstore.similarity_search(inp["description"], k=k)
        retrieved_ids = [r.metadata.get("id") for r in retrieved]

        # compute metrics
        hit = compute_hit_at_k(retrieved_ids, gt, k)
        precision = compute_precision_at_k(retrieved_ids, gt, k)
        recall = compute_recall_at_k(retrieved_ids, gt, k)
        mrr = compute_mrr(retrieved_ids, gt)

        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_scores.append(hit)
        mrr_scores.append(mrr)

        # If no ground truth exists, metrics should reflect that: recall = 0 and hit = 0
        if not gt:
            assert hit == 0, f"Expected no hit when no ground truth for {inp['ticket_id']}"
            assert recall == 0.0, f"Expected recall=0.0 when no ground truth for {inp['ticket_id']}"

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_hit = sum(hit_scores) / len(hit_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)

    print(
        f"\nFull Retrieval Metrics (K={k}): "
        f"Hit@{k}={avg_hit:.2f}, Precision@{k}={avg_precision:.2f}, "
        f"Recall@{k}={avg_recall:.2f}, MRR={avg_mrr:.2f}"
    )

    # Overall metrics assertions
    assert avg_hit >= 0.6, "Average hit@k is too low (<0.2) — retrieval performance may be inadequate"
    assert avg_precision >= 0.5, "Average precision@k is too low (<0.1)"
    assert avg_recall >= 0.5, "Average recall@k is too low (<0.1)"


@pytest.mark.asyncio
async def test_generation_and_format(vectorstore, rag_agent):
    new_inputs = load_new_tickets()
    k = 3
    for inp in new_inputs:
        if inp["ticket_id"] == "TCKT-2008":
            # This ticket has no ground truth, skip generation test
            continue
        res = vectorstore.similarity_search(inp["description"], k=k)
        retrieved_ids = [r.metadata.get("id") for r in res]
        # Build message payload for agent
        payload = {"messages": [{"role": "user", "content": inp["description"]}]}
        response = await rag_agent.ainvoke(payload, {"configurable": {"thread_id": f"eval-{inp['ticket_id']}"}})
        text = response["messages"][-1].content

        # Check format
        assert "Referenced Ticket IDs:" in text, f"Agent response missing reference list for {inp['ticket_id']}"

        # Extract referenced ids
        part = (text.split("Referenced Ticket IDs:")[-1]).split("\n")[0].strip()
        ref_ids = [x.strip().strip("[]") for x in part.strip("[]").split(",") if x.strip()]

        # At least one referenced id from retrieved_ids
        assert any(rid in ref_ids for rid in retrieved_ids), \
            f"Agent did not reference retrieved IDs for {inp['ticket_id']}. retrieved={retrieved_ids}, refs={ref_ids}"
        
        # Check that the suggestion is not just the query echoed
        assert inp["description"] not in text, f"Agent appears to only repeat query for {inp['ticket_id']}"

