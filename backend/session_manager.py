from threading import Lock
from langchain_classic.memory import ConversationBufferWindowMemory
import uuid

class SessionManager:
    def __init__(self, max_sessions=100):
        self.sessions = {}
        self.lock = Lock()
        self.max_sessions = max_sessions
    
    def get_or_create_session(self, session_id: str):
        with self.lock:
            if session_id not in self.sessions:
                if len(self.sessions) >= self.max_sessions:
                    # Remove oldest session
                    oldest = next(iter(self.sessions))
                    del self.sessions[oldest]
                
                self.sessions[session_id] = ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=3
                )
            
            return self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

# Global instance
session_manager = SessionManager()