# tests/multiturn_test.py

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import embeddings, llm
from memory import get_history, save_message, clear_history

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

REWRITE_PROMPT = PromptTemplate.from_template("""
Given the conversation history below and a follow-up question, rewrite the follow-up question into a clear standalone question.

Rules:
- If already standalone, return as-is.
- If it refers to something from history, rewrite using the actual topic.
- If completely new topic, return as-is.
- Return ONLY the rewritten question, nothing else.

Conversation History:
{history}

Follow-up Question:
{question}

Standalone Question:
""")


def format_history(history):
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def answer_with_rewrite(question: str, session_id: str) -> dict:
    """Answer question with visible rewriting step."""

    # Get history
    history = get_history(session_id)

    # Rewrite question
    if history:
        history_str = format_history(history)
        rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = rewrite_chain.invoke({
            "history": history_str,
            "question": question
        }).strip()
    else:
        rewritten = question

    # Retrieve chunks
    vectorstore = Chroma(
        collection_name=f"session_{session_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(rewritten)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate answer
    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "context": context,
        "question": rewritten
    })

    # Save to history
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return {
        "original": question,
        "rewritten": rewritten,
        "answer": answer
    }


# Test conversation
conversation = [
    # Phase 1: Normal follow-ups
    {"phase": "Phase 1 — Normal Follow-ups", "question": "What is DFT?"},
    {"phase": None, "question": "What are its main benefits?"},
    {"phase": None, "question": "What are the challenges?"},
    {"phase": None, "question": "How does AI help with that?"},

    # Phase 2: Topic switch
    {"phase": "Phase 2 — Topic Switch", "question": "What is BIST?"},
    {"phase": None, "question": "How does AI enhance it?"},

    # Phase 3: Return to earlier topic
    {"phase": "Phase 3 — Return to Earlier Topic",
     "question": "Going back to the challenges we discussed, what solutions exist?"},
]


def run_multiturn_test():
    # Clear history for fresh test
    clear_history(SESSION_ID)

    print("\n" + "=" * 65)
    print("MULTI-TURN ROBUSTNESS TEST")
    print("=" * 65)

    for i, turn in enumerate(conversation):
        if turn["phase"]:
            print(f"\n--- {turn['phase']} ---\n")

        print(f"Q{i + 1}: {turn['question']}")

        try:
            result = answer_with_rewrite(turn["question"], SESSION_ID)

            # Show rewriting if question was changed
            if result["rewritten"] != turn["question"]:
                print(f"🔄 Rewritten as: {result['rewritten']}")
            else:
                print(f"🔄 No rewriting needed")

            print(f"A: {result['answer'][:200]}...")

        except Exception as e:
            print(f"ERROR: {str(e)[:150]}")

        print()

        # Wait between calls to avoid rate limits
        if i < len(conversation) - 1:
            print("Waiting 10 seconds...")
            time.sleep(15)


if __name__ == "__main__":
    run_multiturn_test()