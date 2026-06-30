# backend/retrieval.py

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import embeddings, llm, classifier_llm
from memory import get_history, save_message
import json

CHROMA_DIR = "./chroma_db"
TOP_K = 5

# ============================================================
# INTENT CLASSIFICATION — single focused job, uses bigger model
# ============================================================

CLASSIFY_PROMPT = PromptTemplate.from_template("""
Classify the user's message into exactly ONE category. Respond with ONLY valid JSON, nothing else — no markdown, no explanation.

Categories:
- "GREETING_SMALLTALK": greetings, thanks, farewells, casual chit-chat, acknowledgments like "ok"/"good"/"yes"/"no", vague questions like "whats this"/"what's up", or sharing personal info like "my name is X"
- "ABUSE": insults, profanity, rude language directed at the assistant
- "UNCLEAR": ONLY for input that is pure punctuation, random single characters with no words, or truly incomprehensible gibberish. If the message contains any real words forming a sentence, it is NOT unclear.
- "DOCUMENT_QUESTION": a real question seeking facts/explanation/summary/rating FROM the uploaded document itself
- "META_REQUEST": asking the assistant to generate creative content (speech/story/poem) based on the document, OR asking about/rating/critiquing content the assistant itself generated earlier in this conversation (not the document)

Conversation History (most recent last):
{history}

Last content type the assistant generated (if any): {last_generated_type}

User's New Message: "{question}"

Respond with ONLY this JSON structure:
{{"category": "ONE_OF_THE_CATEGORIES_ABOVE"}}
""")

REWRITE_PROMPT = PromptTemplate.from_template("""
Rewrite the user's message into a clear, standalone question using the conversation history to resolve any vague references like "it", "that", "this".

Important rules:
- If the user explicitly says "the pdf" or "the document", that ALWAYS means the original uploaded document — never the assistant's own previously generated content (like a speech or story).
- If the user says "the speech", "that", or "it" in a context where the assistant just generated creative content, that refers to the assistant's generated content, not the document.
- If already standalone and clear, return it as-is.
- Return ONLY the rewritten question, nothing else.

Conversation History:
{history}

Message: {question}

Standalone Question:
""")

RAG_PROMPT = PromptTemplate.from_template("""
You are a friendly, knowledgeable assistant helping users understand a specific document. Respond naturally — not like a rigid FAQ bot.

Guidelines:
- Match response length to the question. Simple questions get concise answers, complex ones get detailed answers.
- For summary requests, give a THOROUGH summary covering ALL major sections — at least 150-200 words.
- If asked to rate or evaluate the document, give an actual rating immediately using the exact scale requested.
- If the answer is in the context, answer naturally using that information.
- If the answer is NOT in the context, say so briefly and naturally.
- Use bullet points only when genuinely useful for multiple distinct points.
- Be warm and conversational, like ChatGPT or Claude — not robotic.

Context:
{context}

Question:
{question}

Answer:
""")

META_PROMPT = PromptTemplate.from_template("""
You are a friendly assistant. The user is asking about content you previously generated, or requesting new creative content based on the document.

Conversation History:
{history}

Document Context (use if generating new creative content):
{context}

User's Request:
{question}

Respond naturally and helpfully, referring to your previous generated content directly if relevant.
""")

SMALLTALK_PROMPT = PromptTemplate.from_template("""
You are a warm, friendly document assistant for an uploaded PDF document. Respond briefly and naturally to this casual or vague message — 1-3 sentences. If the message is vague (like "whats this" or "what's up"), gently invite them to ask about the document.

Conversation History:
{history}

User's Message: {question}

Response:
""")

ABUSE_PROMPT = PromptTemplate.from_template("""
The user sent a rude or abusive message. Respond calmly, briefly, and professionally — redirect to helping with the document without lecturing or being defensive. 1-2 sentences max.

User's Message: {question}

Response:
""")

def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:150]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def get_vectorstore(session_id: str) -> Chroma:
    return Chroma(
        collection_name=f"session_{session_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

def format_chunks(docs) -> str:
    return "\n\n".join([doc.page_content for doc in docs])

def classify_intent(question: str, history: list[dict], last_generated_type: str) -> str:
    history_str = format_history(history)
    chain = CLASSIFY_PROMPT | classifier_llm | StrOutputParser()
    raw = chain.invoke({
        "history": history_str,
        "question": question,
        "last_generated_type": last_generated_type or "None"
    })

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        return result.get("category", "DOCUMENT_QUESTION")
    except (json.JSONDecodeError, AttributeError):
        return "DOCUMENT_QUESTION"

def answer_question(question: str, session_id: str, last_generated_type: str = None) -> dict:
    history = get_history(session_id)

    try:
        category = classify_intent(question, history, last_generated_type)
    except Exception:
        category = "DOCUMENT_QUESTION"

    history_str = format_history(history)

    if category == "GREETING_SMALLTALK":
        chain = SMALLTALK_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"history": history_str, "question": question})
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        return {"answer": answer, "num_chunks_used": 0, "generated_type": "conversational"}

    if category == "ABUSE":
        chain = ABUSE_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        return {"answer": answer, "num_chunks_used": 0, "generated_type": "conversational"}

    if category == "UNCLEAR":
        answer = "I didn't quite catch that — could you rephrase your question?"
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        return {"answer": answer, "num_chunks_used": 0, "generated_type": "conversational"}

    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    try:
        standalone_question = rewrite_chain.invoke({"history": history_str, "question": question}).strip()
    except Exception:
        standalone_question = question

    vectorstore = get_vectorstore(session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(standalone_question)
    context = format_chunks(docs)

    if category == "META_REQUEST":
        chain = META_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"history": history_str, "context": context, "question": standalone_question})
        generated_type = "meta_content"
    else:
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": standalone_question})
        generated_type = "document_answer"

    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return {"answer": answer, "num_chunks_used": len(docs), "generated_type": generated_type}

def is_multiple_questions(question: str) -> bool:
    question_marks = question.count("?")
    if question_marks < 2:
        return False
    parts = [p.strip() for p in question.split("?") if p.strip()]
    return len(parts) >= 2