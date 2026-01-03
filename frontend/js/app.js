// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    API_ENDPOINT: '/api/v1/chat/sync',
    MESSAGE_BATCH_SIZE: 10,
    MAX_MESSAGE_LENGTH: 500,
    RETRY_ATTEMPTS: 3,
    TIMEOUT_MS: 600000
};

// State Management
const state = {
    isConnected: false,
    isLoading: false,
    sessionId: localStorage.getItem('sessionId') || `user_${Date.now()}`,
    messages: JSON.parse(localStorage.getItem('chatHistory') || '[]'),
    lastActivity: Date.now()
};

// DOM Elements
const elements = {
    messagesArea: document.getElementById('messagesArea'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    loadingIndicator: document.getElementById('loadingIndicator'),
    errorBanner: document.getElementById('errorBanner'),
    connectionStatus: document.getElementById('connectionStatus'),
    welcomeTimestamp: document.getElementById('welcomeTimestamp')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    elements.welcomeTimestamp.textContent = formatTimestamp(Date.now());
    loadChatHistory();
    setupEventListeners();
    checkConnection();
    elements.messageInput.focus();
    localStorage.setItem('sessionId', state.sessionId);
}

function setupEventListeners() {
    // Send button click
    elements.sendButton.addEventListener('click', (e) => {
        e.preventDefault();
        sendMessage();
    });

    // Input field events
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    elements.messageInput.addEventListener('input', (e) => {
        const value = e.target.value;
        if (value.length > CONFIG.MAX_MESSAGE_LENGTH) {
            showError(`Message too long (max ${CONFIG.MAX_MESSAGE_LENGTH} chars)`);
            e.target.value = value.slice(0, CONFIG.MAX_MESSAGE_LENGTH);
        }
        updateSendButton();
    });

    // Periodic checks
    setInterval(checkConnection, 5000);
    setInterval(() => {
        if (Date.now() - state.lastActivity > 30 * 60 * 1000) {
            clearChatHistory();
        }
    }, 60000);
}

async function checkConnection() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        const isConnected = response.ok;
        updateConnectionStatus(isConnected);
        state.isConnected = isConnected;
    } catch (error) {
        console.error('Connection check failed:', error);
        updateConnectionStatus(false);
        state.isConnected = false;
    }
}

// 1. Create a variable outside the function to track the LAST known state
let lastStatus = null; 
let statusTimeout;

function updateConnectionStatus(isConnected) {
    const status = elements.connectionStatus;
    
    // 2. NEW LOGIC: If the status hasn't actually changed, DO NOTHING.
    // This stops the constant "re-appearing" loop.
    if (isConnected === lastStatus) return;

    // Update the tracker
    lastStatus = isConnected;

    // Clear any old timers
    if (statusTimeout) clearTimeout(statusTimeout);

    if (isConnected) {
        status.textContent = 'Connected';
        status.className = 'connection-status connected';
        status.style.opacity = '1';

        // Hide after 3 seconds and NEVER show again unless we disconnect first
        statusTimeout = setTimeout(() => {
            status.style.opacity = '0';
            status.style.pointerEvents = 'none';
        }, 3000);
    } else {
        // If disconnected, show it and KEEP it visible
        status.textContent = 'Disconnected';
        status.className = 'connection-status disconnected';
        status.style.opacity = '1';
        status.style.pointerEvents = 'auto';
    }
}



async function sendMessage() {
    const question = elements.messageInput.value.trim();
    console.log('Sending message:', question); // Debug log

    if (!question || state.isLoading) {
        console.log('Message not sent - empty or loading:', {question, isLoading: state.isLoading});
        return;
    }

    if (!state.isConnected) {
        showError('Not connected to server. Please check your connection.');
        return;
    }

    hideError();
    addMessageToUI(question, 'user');
    elements.messageInput.value = '';
    updateSendButton();
    showLoading();

    await callSupportAgent(question);
}

async function callSupportAgent(question) {
    const payload = {
        question: question,
        session_id: state.sessionId
    };

    console.log('Sending request to:', `${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINT}`); // Debug log
    console.log('Payload:', payload); // Debug log

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CONFIG.TIMEOUT_MS);

        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.API_ENDPOINT}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('Response received:', result); // Debug log
        addMessageToUI(result.answer, 'bot', result.sources);
        saveToHistory(question, result.answer);

    } catch (error) {
        console.error('API call failed:', error);
        const errorMessage = 'Sorry, I encountered an error. Please try again.';
        addMessageToUI(errorMessage, 'bot');
        showError(errorMessage);
    } finally {
        hideLoading();
        state.lastActivity = Date.now();
    }
}

function addMessageToUI(content, sender, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'message-timestamp';
    timestampDiv.textContent = formatTimestamp(Date.now());

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timestampDiv);

    if (sender === 'bot' && sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.textContent = `Source: ${sources.join(', ')}`;
        messageDiv.appendChild(sourcesDiv);
    }

    if (sender === 'bot') {
    console.log("Bot message detected, adding buttons..."); // DEBUG LOG
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = 'message-feedback';

    const thumbsUp = document.createElement('button');
    thumbsUp.type = 'button'; // Explicitly set type
    thumbsUp.innerHTML = '👍';
        thumbsUp.title = 'Helpful';
        thumbsUp.onclick = () => {
            thumbsUp.classList.add('clicked');
            thumbsDown.disabled = true;
            recordFeedback('thumbs_up', state.sessionId);
        };

        const thumbsDown = document.createElement('button');
        thumbsDown.innerHTML = '👎';
        thumbsDown.title = 'Not helpful';
        thumbsDown.onclick = () => {
            thumbsDown.classList.add('clicked');
            thumbsUp.disabled = true;
            recordFeedback('thumbs_down', state.sessionId);
        };

        feedbackDiv.appendChild(thumbsUp);
        feedbackDiv.appendChild(thumbsDown);
        messageDiv.appendChild(feedbackDiv);
    }

    elements.messagesArea.appendChild(messageDiv);
    scrollToBottom();
}

async function recordFeedback(rating, sessionId) {
    try {
        console.log('Recording feedback:', {rating, sessionId}); // Debug log
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/v1/feedback`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ rating, session_id: sessionId })
        });

        if (!response.ok) {
            console.error('Feedback recording failed:', response.status);
        } else {
            console.log('Feedback recorded successfully:', rating);
        }
    } catch (e) {
        console.error('Feedback error:', e);
    }
}

function formatTimestamp(ts) {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
    elements.messagesArea.scrollTop = elements.messagesArea.scrollHeight;
}

function showLoading() {
    state.isLoading = true;
    elements.loadingIndicator.classList.add('active');
    elements.loadingIndicator.textContent = "Agent is thinking...";
    elements.messageInput.disabled = true;
    elements.sendButton.disabled = true;
    updateSendButton(); // Ensure button state is updated
}

function hideLoading() {
    state.isLoading = false;
    elements.loadingIndicator.classList.remove('active');
    elements.messageInput.disabled = false;
    elements.sendButton.disabled = false;
    updateSendButton(); // Ensure button state is updated
    elements.messageInput.focus();
}

function updateSendButton() {
    const hasText = elements.messageInput.value.trim() !== '';
    const canSend = hasText && !state.isLoading && state.isConnected;
    elements.sendButton.disabled = !canSend;
    console.log('Button state updated:', {hasText, isLoading: state.isLoading, isConnected: state.isConnected, canSend}); // Debug log
}

function showError(msg) {
    console.error('Error:', msg); // Debug log
    elements.errorBanner.textContent = msg;
    elements.errorBanner.classList.add('active');
    setTimeout(hideError, 5000);
}

function hideError() {
    elements.errorBanner.classList.remove('active');
}

function saveToHistory(q, a) {
    state.messages.push({ question: q, answer: a, timestamp: Date.now() });
    if (state.messages.length > 50) state.messages.shift();
    localStorage.setItem('chatHistory', JSON.stringify(state.messages));
}

function loadChatHistory() {
    state.messages.slice(-10).forEach(m => {
        addMessageToUI(m.question, 'user');
        addMessageToUI(m.answer, 'bot');
    });
}

function clearChatHistory() {
    localStorage.clear();
    location.reload();
}