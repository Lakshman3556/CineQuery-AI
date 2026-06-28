from langchain_classic.memory import ConversationBufferWindowMemory
from typing import Dict

class SessionMemoryManager:
    """
    Manages in-memory conversation history using LangChain's 
    ConversationBufferWindowMemory on a per-session basis.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.sessions: Dict[str, ConversationBufferWindowMemory] = {}

    def get_session_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Retrieves or creates a ConversationBufferWindowMemory instance for a session.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferWindowMemory(
                k=self.k, 
                memory_key="history",
                input_key="question",
                output_key="answer"
            )
        return self.sessions[session_id]

    def load_history_string(self, session_id: str) -> str:
        """
        Loads the history variables as a formatted string.
        """
        mem = self.get_session_memory(session_id)
        vars = mem.load_memory_variables({})
        return vars.get("history", "")

    def add_turn(self, session_id: str, question: str, answer: str):
        """
        Appends a user question and model response to the session memory.
        """
        mem = self.get_session_memory(session_id)
        mem.save_context(
            {"question": question},
            {"answer": answer}
        )

# Global memory manager instance
memory_manager = SessionMemoryManager(k=5)
