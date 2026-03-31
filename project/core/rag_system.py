import uuid
from langchain_openrouter import ChatOpenRouter
import config
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph

class RAGSystem:
    
    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = 50
        
    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing or empty (after trimming whitespace). "
                "Set it in project/.env or use OPENROUTER_KEY; see https://openrouter.ai/settings/keys"
            )

        llm_kwargs: dict = {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "api_key": config.OPENROUTER_API_KEY,
        }
        if config.OPENROUTER_APP_URL:
            llm_kwargs["app_url"] = config.OPENROUTER_APP_URL
        if config.OPENROUTER_APP_TITLE:
            llm_kwargs["app_title"] = config.OPENROUTER_APP_TITLE
        if config.OPENROUTER_API_BASE:
            llm_kwargs["base_url"] = config.OPENROUTER_API_BASE

        llm = ChatOpenRouter(**llm_kwargs)
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)
        
    def get_config(self):
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
    
    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())