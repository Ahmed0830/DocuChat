"""
RAGAS evaluation script — baseline vs re-ranked RAG pipeline.

Usage
-----
    cd backend
    uv run python evaluate.py

Requirements
------------
- PDFs must be present in data/pdfs/ before running.
- DIAL_API_KEY, DIAL_ENDPOINT, DIAL_API_VERSION, DIAL_DEPLOYMENT must be set
  in a .env file in the backend directory (same as the FastAPI app).

What it does
------------
1. Loads all PDFs from data/pdfs/ and ingests them into a shared in-memory
   Qdrant collection.
2. Runs a curated set of test questions through two RAGSearch instances:
     - baseline  : use_reranker=False  (plain ANN retrieval)
     - reranked   : use_reranker=True   (ANN + CrossEncoder re-ranking)
3. Evaluates both result sets with three reference-free RAGAS metrics:
     - Faithfulness               – answer grounded in context?
     - ResponseRelevancy          – answer on-topic for the question?
     - LLMContextPrecisionWithoutReference – retrieved chunks useful?
4. Prints a side-by-side comparison table and saves:
     results_baseline.json
     results_reranked.json
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── RAGAS imports ──────────────────────────────────────────────────────────────
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

# ── Local project imports ──────────────────────────────────────────────────────
from rag.data_loader import load_from_paths
from rag.llm import get_llm
from rag.search import RAGSearch
from rag.vectorstore import QdrantStore

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_DIR = Path(__file__).parent / "data" / "pdfs"
RETRIEVAL_K = 6

# Hand-curated evaluation questions covering the PDFs in data/pdfs/.
# Easier to maintain than auto-generated testsets which can produce 0 questions
# on cheatsheet-style documents.
QUESTIONS: list[str] = [
    "What are the main types of embedding models used in NLP?",
    "How does a bi-encoder differ from a cross-encoder for embeddings?",
    "What is Retrieval-Augmented Generation and how does it work?",
    "What chunking strategies are recommended for RAG pipelines?",
    "How can you evaluate the quality of a RAG system?",
    "What are Python list comprehensions and how do you use them?",
    "Explain Python decorators with an example.",
    "What are the key differences between lists and tuples in Python?",
]


def _collect_pdfs() -> list[str]:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}. "
            "Add at least one PDF before running evaluation."
        )
    print(f"[+] Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")
    return [str(p) for p in pdfs]


def _build_shared_store(embeddings: HuggingFaceEmbeddings) -> QdrantStore:
    """Ingest all PDFs into a single in-memory Qdrant collection."""
    store = QdrantStore(
        collection_name="eval_collection",
        location=":memory:",
        embeddings=embeddings,
    )
    paths = _collect_pdfs()
    docs = load_from_paths(paths)
    print(f"[+] Loaded {len(docs)} document page(s). Ingesting …")
    chunk_count = store.build_from_documents(docs)
    print(f"[+] Ingested {chunk_count} chunk(s).")
    return store


def _build_eval_dataset(
    rag: RAGSearch,
    questions: list[str],
    label: str,
) -> EvaluationDataset:
    """Run every question through *rag* and build a RAGAS EvaluationDataset."""
    print(f"[+] Running inference — {label} …")
    samples: list[SingleTurnSample] = []
    for i, question in enumerate(questions, 1):
        print(f"    [{i}/{len(questions)}] {question[:80]}")
        docs = rag.retrieve(question, k=RETRIEVAL_K)
        contexts = [doc.page_content for doc in docs]
        answer = rag.search(question, k=RETRIEVAL_K, session_id=f"{label}_{i}")
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
        )
    return EvaluationDataset(samples=samples)


def _run_ragas(
    dataset: EvaluationDataset,
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: LangchainEmbeddingsWrapper,
) -> dict:
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )
    return result


def _print_comparison(baseline: dict, reranked: dict) -> None:
    metric_keys = [
        "faithfulness",
        "response_relevancy",
        "llm_context_precision_without_reference",
    ]
    col_w = 12
    header = f"{'Metric':<45} {'Baseline':>{col_w}} {'Re-ranked':>{col_w}} {'Delta':>{col_w}}"
    print("\n" + "=" * len(header))
    print("RAGAS EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key in metric_keys:
        b = baseline.get(key, float("nan"))
        r = reranked.get(key, float("nan"))
        delta = r - b if isinstance(b, float) and isinstance(r, float) else float("nan")
        delta_str = f"{delta:+.4f}" if not (delta != delta) else "  N/A"
        label = key.replace("_", " ").title()
        print(f"{label:<45} {b:>{col_w}.4f} {r:>{col_w}.4f} {delta_str:>{col_w}}")
    print("=" * len(header) + "\n")


def main() -> None:
    # ── Shared embedding model ─────────────────────────────────────────────────
    print("[+] Loading embedding model …")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ── LLM + RAGAS wrappers ───────────────────────────────────────────────────
    print("[+] Initialising LLM …")
    llm = get_llm()
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    # ── Shared vector store ────────────────────────────────────────────────────
    shared_store = _build_shared_store(embeddings)

    # ── Two RAGSearch instances sharing the same store ─────────────────────────
    baseline_rag = RAGSearch(vectorstore=shared_store, llm=llm, use_reranker=False)
    reranked_rag = RAGSearch(vectorstore=shared_store, llm=llm, use_reranker=True)

    # ── Test questions ─────────────────────────────────────────────────────────
    questions = QUESTIONS
    print(f"[+] Using {len(questions)} curated evaluation question(s).")

    # ── Build evaluation datasets ──────────────────────────────────────────────
    baseline_dataset = _build_eval_dataset(baseline_rag, questions, label="baseline")
    reranked_dataset = _build_eval_dataset(reranked_rag, questions, label="reranked")

    # ── Run RAGAS evaluation ───────────────────────────────────────────────────
    print("[+] Evaluating baseline …")
    baseline_scores = _run_ragas(baseline_dataset, ragas_llm, ragas_emb)

    print("[+] Evaluating re-ranked …")
    reranked_scores = _run_ragas(reranked_dataset, ragas_llm, ragas_emb)

    # ── Print comparison table ─────────────────────────────────────────────────
    _print_comparison(baseline_scores._repr_dict, reranked_scores._repr_dict)

    # ── Save raw results ───────────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    baseline_path = out_dir / "results_baseline.json"
    reranked_path = out_dir / "results_reranked.json"

    baseline_path.write_text(
        json.dumps(
            {"scores": baseline_scores._repr_dict, "samples": baseline_dataset.to_list()},
            indent=2,
            default=str,
        )
    )
    reranked_path.write_text(
        json.dumps(
            {"scores": reranked_scores._repr_dict, "samples": reranked_dataset.to_list()},
            indent=2,
            default=str,
        )
    )
    print(f"[+] Saved {baseline_path.name} and {reranked_path.name}")


if __name__ == "__main__":
    main()