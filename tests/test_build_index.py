import json
import pandas as pd
from pathlib import Path
import pytest

from src.build_index import (
    load_dataframe_from_file,
    load_json_file,
    row_to_doc,
    deduplicate_documents,
    DATA_DIR,
    PERSIST_DIR,
    COLLECTION,
    MODEL_NAME,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


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


def write_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def write_xlsx(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)
    return path


def write_json_dict(tmp_path, dict_format):
    path = tmp_path / "test.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict_format, f)
    return path


def test_load_dataframe_from_csv(temp_dir):
    rows = [{"Ticket ID": "T1", "Issue": "Issue1", "Description": "Desc1", "Resolution": "Res1"}]
    path = write_csv(temp_dir, rows)
    df = load_dataframe_from_file(path)
    assert len(df) == 1
    assert df.loc[0, "Ticket ID"] == "T1"
    assert df.loc[0, "Issue"] == "Issue1"


def test_load_dataframe_from_xlsx(temp_dir):
    rows = [{"Ticket ID": "T2", "Issue": "Issue2", "Description": "Desc2", "Resolution": "Res2"}]
    path = write_xlsx(temp_dir, rows)
    df = load_dataframe_from_file(path)
    assert "Ticket ID" in df.columns
    assert df.iloc[0]["Issue"] == "Issue2"


def test_load_json_file(temp_dir):
    # Build JSON dict-format like your example
    dict_format = {
        "Ticket ID": {"0": "T3"},
        "Issue": {"0": "Issue3"},
        "Description": {"0": "Desc3"},
        "Resolution": {"0": "Res3"},
        "Category": {"0": "Cat3"},
        "Date": {"0": "2024-01-01"},
        "Agent Name": {"0": "Agent3"},
    }
    path = write_json_dict(temp_dir, dict_format)
    rows = load_json_file(path)
    assert isinstance(rows, list)
    assert len(rows) == 1
    row0 = rows[0]
    assert row0["Ticket ID"] == "T3"
    assert row0["Issue"] == "Issue3"
    assert row0["Resolution"] == "Res3"


def test_row_to_doc_and_metadata():
    row = {"Ticket ID": "T4", "Issue": "Issue4", "Description": "Desc4", "Resolution": "Res4", "Category": "Cat4", "Date": "2024-01-02", "Agent Name": "Agent4"}
    doc = row_to_doc(row)
    assert doc.page_content.startswith("Title: Issue4")
    assert doc.metadata["id"] == "T4"
    assert doc.metadata["agent"] == "Agent4"


def test_deduplicate_documents():
    docs = [
        row_to_doc({"Ticket ID": "T5", "Issue": "I1", "Description": "", "Resolution": ""}),
        row_to_doc({"Ticket ID": "T5", "Issue": "I1_dup", "Description": "", "Resolution": ""}),
        row_to_doc({"Ticket ID": "T6", "Issue": "I2", "Description": "", "Resolution": ""})
    ]
    unique = deduplicate_documents(docs)
    assert len(unique) == 2
    ids = [d.metadata["id"] for d in unique]
    assert "T5" in ids and "T6" in ids


def test_indexing_document_count(vectorstore, expected_minimum=30):
    all_docs = vectorstore.get(include=["documents"])
    doc_count = len(all_docs["documents"])
    assert doc_count >= expected_minimum, f"Indexed only {doc_count} docs, expected at least {expected_minimum}"


def test_query_vpn_returns_results(vectorstore):
    query = "VPN connection is not working"
    results = vectorstore.similarity_search(query, k=5) 
    assert len(results) >= 2, f"Expected at least 2 results for query '{query}', got {len(results)}"

    ids = [r.metadata.get("id") for r in results]
    expected_ids = {"TCKT-1051", "TKT1002", "TCKT-1011"}
    assert expected_ids.issubset(set(ids)), f"Expected at least {len(expected_ids)} in result ids {ids}"
    
    assert any("VPN" in r.page_content for r in results), "None of top results mention VPN"