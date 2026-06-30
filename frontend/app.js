const API_BASE = "http://127.0.0.1:8000";

let sessionId = null;
let isWaiting = false;

// DOM Elements
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const documentInfo = document.getElementById("documentInfo");
const documentName = document.getElementById("documentName");
const uploadProgress = document.getElementById("uploadProgress");
const chatMessages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const chatStatus = document.getElementById("chatStatus");
const newChatBtn = document.getElementById("newChatBtn");

// ===== Upload Handling =====
uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
});

async function handleFileUpload(file) {
    if (!file.name.endsWith(".pdf")) {
        showError("Only PDF files are allowed.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("File too large. Maximum size is 10MB.");
        return;
    }

    uploadArea.style.display = "none";
    documentInfo.style.display = "none";
    uploadProgress.style.display = "block";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        sessionId = data.session_id;
        uploadProgress.style.display = "none";
        documentInfo.style.display = "flex";
        documentName.textContent = file.name;

        questionInput.disabled = false;
        questionInput.placeholder = "Ask a question about your document...";
        sendBtn.disabled = false;
        chatStatus.textContent = `Chatting about: ${file.name}`;

        clearChat();
        addSystemMessage(`📄 <strong>${file.name}</strong> uploaded successfully! Ask me anything about it.`);

    } catch (error) {
        uploadProgress.style.display = "none";
        uploadArea.style.display = "flex";
        showError(error.message || "Upload failed. Please try again.");
    }
}

// ===== Chat Handling =====
sendBtn.addEventListener("click", sendMessage);

questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + "px";
});

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || isWaiting || !sessionId) return;

    isWaiting = true;
    sendBtn.disabled = true;
    questionInput.disabled = true;
    questionInput.value = "";
    questionInput.style.height = "auto";

    addMessage("user", question);
    const typingId = addTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                session_id: sessionId,
            }),
        });

        const data = await response.json();
        removeTypingIndicator(typingId);

        if (!response.ok) {
            // Show actual error message from backend
            addMessage("ai", `⚠️ ${data.detail || "Something went wrong. Please try again."}`);
            return;
        }

        addMessage("ai", data.answer);

    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage("ai", "⚠️ Could not reach the server. Make sure the backend is running.");
    } finally {
        isWaiting = false;
        sendBtn.disabled = false;
        questionInput.disabled = false;
        questionInput.focus();
    }
}

// ===== Message Rendering =====
function addMessage(role, content) {
    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const avatar = role === "user" ? "👤" : "🤖";

    // Format content — convert newlines and bullet points
    const formatted = formatMessage(content);

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-bubble">${formatted}</div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function formatMessage(text) {
    // Escape HTML first
    const escaped = escapeHtml(text);
    // Convert newlines to <br>
    return escaped.replace(/\n/g, "<br>");
}

function addSystemMessage(content) {
    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.className = "message ai";
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble">${content}</div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function addTypingIndicator() {
    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    const id = "typing-" + Date.now();
    const div = document.createElement("div");
    div.className = "message ai";
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ===== Utility Functions =====
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

function showError(message) {
    const existing = document.querySelector(".error-message");
    if (existing) existing.remove();

    const div = document.createElement("div");
    div.className = "error-message";
    div.textContent = message;
    uploadArea.parentNode.insertBefore(div, uploadArea.nextSibling);

    setTimeout(() => div.remove(), 4000);
}

function clearChat() {
    chatMessages.innerHTML = "";
}

// ===== New Document Button =====
newChatBtn.addEventListener("click", () => {
    sessionId = null;
    uploadArea.style.display = "flex";
    documentInfo.style.display = "none";
    uploadProgress.style.display = "none";
    fileInput.value = "";

    questionInput.disabled = true;
    questionInput.placeholder = "Upload a document first...";
    sendBtn.disabled = false;
    chatStatus.textContent = "Upload a document to begin";

    clearChat();
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🤖</div>
            <h3>Welcome to AI Document Q&A</h3>
            <p>Upload a PDF document from the sidebar to get started. You can then ask any questions about its content.</p>
        </div>
    `;
});