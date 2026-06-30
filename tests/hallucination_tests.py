import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import embeddings, llm

CHROMA_DIR = "../backend/chroma_db"
SESSION_ID = "test123"
TOP_K = 5

RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant that answers questions based ONLY on the provided document context.

Rules:
- If the answer is clearly present in the context, answer it directly and concisely.
- If the answer is NOT in the context, say exactly: "This information is not covered in the uploaded document."
- Do NOT make up information or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""")


def answer_direct(question: str) -> str:
    """Answer without query rewriting — saves 1 API call per question."""
    vectorstore = Chroma(
        collection_name=f"session_{SESSION_ID}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


test_cases = [
    {
        "category": "Out of Scope",
        "question": "What is the price of Synopsys Tessent tool?",
        "expected_behavior": "Should say not covered in document",
        "should_answer": False
    },
    {
        "category": "Out of Scope",
        "question": "Who is the CEO of Cadence Design Systems?",
        "expected_behavior": "Should say not covered in document",
        "should_answer": False
    },
    {
        "category": "Out of Scope",
        "question": "What programming language is used to write DFT tools?",
        "expected_behavior": "Should say not covered in document",
        "should_answer": False
    },
    {
        "category": "Factual",
        "question": "What percentage of chip cost involves testing?",
        "expected_behavior": "Should say ~30%",
        "should_answer": True
    },
    {
        "category": "Factual",
        "question": "What was the ATE test time reduction outcome mentioned in the document?",
        "expected_behavior": "Should say 45% reduction",
        "should_answer": True
    },
    {
        "category": "Factual",
        "question": "What test coverage percentage does AI-Enhanced ATPG achieve?",
        "expected_behavior": "Should say 99%+",
        "should_answer": True
    },
    {
        "category": "Misleading",
        "question": "Since DFT eliminates all chip defects, how does it improve yield?",
        "expected_behavior": "Should not agree that DFT eliminates all defects",
        "should_answer": False
    },
    {
        "category": "Misleading",
        "question": "How does AI reduce chip testing costs to zero?",
        "expected_behavior": "Should not say costs go to zero",
        "should_answer": False
    },
    {
        "category": "Misleading",
        "question": "What did the document say about quantum computing replacing DFT entirely by 2025?",
        "expected_behavior": "Should say not covered or not mentioned in document",
        "should_answer": False
    },
]


def run_hallucination_tests():
    print("\n" + "=" * 65)
    print("HALLUCINATION / FAITHFULNESS TEST RESULTS")
    print("=" * 65)
    print("Review each answer manually and mark PASS or FAIL\n")

    for i, test in enumerate(test_cases):
        print(f"Test {i + 1} [{test['category']}]")
        print(f"Q: {test['question']}")
        print(f"Expected: {test['expected_behavior']}")

        try:
            answer = answer_direct(test['question'])
            print(f"A: {answer}")
        except Exception as e:
            print(f"A: ERROR — {str(e)[:150]}")

        print(f"Should answer: {'YES' if test['should_answer'] else 'NO — should decline'}")
        print("-" * 65)
        print()

        # Wait 8 seconds between calls to stay under 10 RPM
        if i < len(test_cases) - 1:
            print("Waiting 8 seconds before next test...")
            time.sleep(8)


if __name__ == "__main__":
    run_hallucination_tests()