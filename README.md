# 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) assistant for IT help-desk ticket management.  
The goal is to support service-agents by suggesting directions for incoming tickets, using previously resolved tickets as context.  
It is structured as a lightweight app: `embeddings of past tickets` - `vector-store retrieval` - `generation of suggestions`

# 2. Repository Structure

- The project is stored in a **private GitHub repository**
- It uses a virtual environment to isolate dependencies
- Directory layout:
```bash
  ├── src/
  │   └── build_index.py        # Embedding generation & indexing logic
  ├── scripts/
  │   └── app.py                # Application / agent setup
  ├── data/
  │   └── old_tickets/…         # Resolved-tickets CSV/XLSX/JSON data
  ├── tests/                    # Unit-tests & evaluation
  ├── pyproject.toml            # Dependencies
  ├── uv.lock                   # Locked dependencies
  ├── .env                      # Secrets
  └── README.md                 # Overview

  ```

## 3. Environment Setup

#### Virtual Environment & Dependency Management
We use **uv** as our dependency management tool, using`pyproject.toml` for dependency declarations and `uv.lock` to lock exact versions

#### 1. Clone the repository
```bash
git clone https://github.com/AT76/case_study.git
```
#### 2. Install uv 
```bash
pip install uv
```
#### Or using brew:
```bash
brew install uv
```
#### 3. Install dependencies and sync from lockfile
```bash
uv sync           # reads `uv.lock` and installs exact versions
```  

If you update `pyproject.toml`, run:

```bash
uv lock           # updates `uv.lock` to match `pyproject.toml` 
```

Then, activate the env:

```bash
source .venv/bin/activate
```

#### 4. Running the project
- To index old tickets:
    ```bash
    python src/build_index.py
    ```
    
- To launch the application:
    ```bash
    chainlit run scripts/app.py
    ```

## 4. Key Components & Approach

### src/build_index.py
- Loads spreadsheet (CSV/XLSX) or JSON data of resolved tickets via `load_dataframe_from_file` and `load_json_file`
- Converts each row to a `Document` object via `row_to_doc`, including metadata (`id`, `title`, `category`, `date`, `agent`)
- Deduplicates documents by ticket-ID using `deduplicate_documents`
- Embeds documents using `HuggingFaceEmbeddings` and stores them in a `Chroma` vector-store
- Controlled via environment variables (e.g., `OLD_TICKETS_DIR`, `CHROMA_DIR`, `COLLECTION`, `RESET`)

### scripts/app.py
- Builds a retriever from the vector-store (`build_retriever`)
- Builds a chat model endpoint (`build_chat_model`) via `HuggingFaceEndpoint`
- Defines a prompt middleware (`prompt_with_context`) that:
    - Retrieves the top-k similar past tickets for the incoming ticket description
    - Ensures unique IDs (deduplication) in the context
    - Injects the context into a structured system prompt (specifying instructions, referencing past ticket IDs, providing a suggestion)
- Chains retrieval + generation into an interactive app (via Chainlit)
- The UI includes “starters” to ease typical ticket queries

## 5. Assumptions
- We assume that resolved tickets are meaningfully similar to future incoming tickets (e.g., category, issue type)
- We set `k = 3` for retrieval by default, giving the top-3 most similar past tickets
- Suggestions are **directions** rather than full solutions -> the agent still needs to check answers
- The system is intended for internal agents, not end-customers, therefore IT-help-desk knowledge is required
- The user query is clear enough to run a semantic similarity search that will return meaningful results

## 6. Experimentation & Results
- Unit-tests validate data loading, conversion, deduplication, and retrieval behaviour
- Evaluation uses `"new tickets”` data as queries and measures retrieval metrics: `Hit@K`, `Precision@K`, `Recall@K`, `MRR`
- **Early findings**:
    - Retrieval reliably finds “exact match” in past tickets for many query types
    - On some categories (e.g. server access) older tickets are missing -> retrieval fails or suggests generic direction
    - The suggestion generation respects the format (includes “Referenced Ticket IDs: [...]”) and avoids simple echoing of the query
- **Shortcomings observed**:
    - When no close past ticket exists, the context may be weak, leading to less useful suggestions
    - The vector-store may embed documents with similar wording but different root causes, leading to ambiguous context
    - No feedback loop currently captures whether the suggestion actually helped the agent (lack of production usage metric)

## 7. Further Work (if more time)
- Improve metadata filtering: e.g., restrict retrieval to same “Category” to increase relevance
	- Helps avoid retrieving tickets from unrelated categories that might confuse the suggestion
- Introduce reranking of retrieval results (hybrid embedding + keyword matching)
	- Combine semantic embeddings with keyword-matching and then rerank results
	- Reranking can be done using a custom scoring function (cos-sim + keyword-match + metadata-match)
- Add evaluation feedback loop (Positive / Negative answer)
	- Label which retrievals were useful and which were not
	- Can lead to improvement over time
	- Use this in metadata matching
- Enhance the UI
	- Allow agent to see retrieved ticket summaries & click-through to full resolution logs -> Detail Pop-Up
- Provide fallback logic
	- When low similarity score -> Prompt the agent to escalate or seek expert support
	- Low Confidence should be detected (e.g via low similarity score)
- Deploy into production
	- Containerise the app -> Docker
	- Set up CI/CD -> GitHub Actions
	- Enable scalable hosting -> Kubernetes
	- Monitoring and logging of usage + latency -> Grafana, Prometheus