# tests/test_retrieval.py

import json
import sys
import os

# Add backend to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from langchain_chroma import Chroma
from config import embeddings

CHROMA_DIR = "../backend/chroma_db"
SESSION_ID = "test123"
TOP_K = 5


def get_all_chunks(session_id: str) -> list[str]:
    """Get all chunks from ChromaDB for a session."""
    vectorstore = Chroma(
        collection_name=f"session_{session_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    result = vectorstore.get()
    return result['documents']


def retrieve_chunks(question: str, session_id: str, k: int) -> list[int]:
    """Retrieve top-k chunks and return their indices."""
    vectorstore = Chroma(
        collection_name=f"session_{session_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    # Get all chunks to build index map
    all_docs = vectorstore.get()
    all_texts = all_docs['documents']

    # Retrieve top-k
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)

    # Find indices of retrieved chunks
    retrieved_indices = []
    for doc in retrieved_docs:
        for i, text in enumerate(all_texts):
            if text == doc.page_content:
                retrieved_indices.append(i)
                break

    return retrieved_indices


def precision_at_k(retrieved: list[int], relevant: list[int]) -> float:
    """What fraction of retrieved chunks were relevant?"""
    if not retrieved:
        return 0.0
    hits = len(set(retrieved) & set(relevant))
    return hits / len(retrieved)


def recall_at_k(retrieved: list[int], relevant: list[int]) -> float:
    """What fraction of relevant chunks were retrieved?"""
    if not relevant:
        return 0.0
    hits = len(set(retrieved) & set(relevant))
    return hits / len(relevant)


def run_evaluation(session_id: str, k: int = 5):
    """Run full retrieval evaluation against test set."""

    # Load test set
    test_set_path = os.path.join(os.path.dirname(__file__), 'test_set.json')
    with open(test_set_path, 'r') as f:
        test_set = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL EVALUATION — k={k}, session={session_id}")
    print(f"{'=' * 60}\n")

    precisions = []
    recalls = []

    for i, test_case in enumerate(test_set):
        question = test_case['question']
        relevant = test_case['relevant_chunks']

        retrieved = retrieve_chunks(question, session_id, k)

        p = precision_at_k(retrieved, relevant)
        r = recall_at_k(retrieved, relevant)

        precisions.append(p)
        recalls.append(r)

        # Per question result
        hit = "✅" if len(set(retrieved) & set(relevant)) > 0 else "❌"
        print(f"Q{i + 1}: {question[:55]}...")
        print(f"     Relevant: {relevant}")
        print(f"     Retrieved: {retrieved}")
        print(f"     Precision@{k}: {p:.2f} | Recall@{k}: {r:.2f} {hit}")
        print()

    # Summary
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    print(f"{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Average Precision@{k}: {avg_precision:.2f} ({avg_precision * 100:.1f}%)")
    print(f"Average Recall@{k}:    {avg_recall:.2f} ({avg_recall * 100:.1f}%)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_evaluation(session_id="test123", k=5)