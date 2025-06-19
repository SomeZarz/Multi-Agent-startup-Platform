import os
import yaml
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

class RAGKnowledgeManager:
    def __init__(self, config_path="knowledge_system/config/kb_config.yaml"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file not found at {config_path}. Using default configuration.")
            self.config = self._get_default_config()
        
        self.embeddings = OpenAIEmbeddings(
            model=self.config.get('embedding_model', 'text-embedding-3-small'),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load knowledge bases with RAG capabilities
        self.knowledge_bases = self._load_all_knowledge_bases()
    
    def _get_default_config(self):
        """Default configuration if config file is missing"""
        return {
            "embedding_model": "text-embedding-3-small",
            "agents": {
                "CEO": {"max_retrieval_results": 3, "relevance_threshold": 0.7},
                "CFO": {"max_retrieval_results": 3, "relevance_threshold": 0.7},
                "CTO": {"max_retrieval_results": 3, "relevance_threshold": 0.7},
                "COO": {"max_retrieval_results": 3, "relevance_threshold": 0.7}
            }
        }
    
    def _load_all_knowledge_bases(self) -> Dict[str, FAISS]:
        """Load all knowledge bases with RAG support and proper security settings"""
        knowledge_bases = {}
        
        kb_paths = {
            "CEO": "knowledge_system/vector_stores/ceo_market_db",
            "CFO": "knowledge_system/vector_stores/cfo_funding_db",
            "CTO": "knowledge_system/vector_stores/cto_tech_db",
            "COO": "knowledge_system/vector_stores/coo_operations_db"
        }
        
        for agent, path in kb_paths.items():
            try:
                if os.path.exists(path):
                    # FIX: Add allow_dangerous_deserialization=True for your own files
                    knowledge_bases[agent] = FAISS.load_local(
                        path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True  # Safe since we created these files
                    )
                    print(f"✅ RAG-enabled {agent} knowledge base loaded")
                else:
                    # Create empty knowledge base
                    dummy_doc = Document(page_content="No knowledge available", metadata={"source": "empty"})
                    knowledge_bases[agent] = FAISS.from_documents([dummy_doc], self.embeddings)
                    print(f"⚠️ Created empty {agent} knowledge base")
            except Exception as e:
                print(f"❌ Error loading {agent} knowledge base: {e}")
                # Create fallback knowledge base
                dummy_doc = Document(page_content="Knowledge base error", metadata={"source": "error"})
                knowledge_bases[agent] = FAISS.from_documents([dummy_doc], self.embeddings)
        
        return knowledge_bases
    
    def rag_retrieve_and_rank(self, agent_type: str, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """RAG retrieval with similarity scores and ranking"""
        if agent_type not in self.knowledge_bases:
            return []
        
        try:
            # Get documents with similarity scores
            docs_and_scores = self.knowledge_bases[agent_type].similarity_search_with_score(
                query, k=k
            )
            
            # Filter by relevance threshold
            agent_config = self.config['agents'].get(agent_type, {})
            threshold = agent_config.get('relevance_threshold', 0.7)
            
            # Filter out error documents and apply threshold
            filtered_docs = []
            for doc, score in docs_and_scores:
                content = doc.page_content.strip()
                if (score <= threshold and  # Lower scores = higher similarity in FAISS
                    content and 
                    "error" not in content.lower() and
                    "no knowledge available" not in content.lower() and
                    len(content) > 20):
                    filtered_docs.append((doc, score))
            
            return filtered_docs
            
        except Exception as e:
            print(f"❌ RAG retrieval error for {agent_type}: {e}")
            return []
    
    def rag_generate_context(self, agent_type: str, business_context: str, query: str) -> Dict[str, any]:
        """Generate RAG context for agent with proper formatting"""
        
        # Step 1: Query expansion for better retrieval
        expanded_query = f"{business_context} {query} {agent_type.lower()} expertise startup business"
        
        # Step 2: Retrieve relevant documents with scores
        retrieved_docs = self.rag_retrieve_and_rank(agent_type, expanded_query, k=3)
        
        if not retrieved_docs:
            return {
                "context": "",
                "sources": [],
                "relevance_scores": [],
                "retrieval_success": False
            }
        
        # Step 3: Format context in agent's natural style
        context_parts = []
        sources = []
        scores = []
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get('source', 'unknown')
            
            # Clean and extract key insights
            lines = content.split('\n')
            key_insights = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('-') and len(line) > 15:
                    # Remove bullet points and clean up
                    cleaned_line = line.replace('•', '').replace('*', '').replace('-', '').strip()
                    if cleaned_line and len(cleaned_line) > 10:
                        key_insights.append(cleaned_line)
            
            if key_insights:
                # Take the most relevant insight
                context_parts.append(key_insights[0])
                sources.append(source)
                scores.append(float(score))
        
        # Step 4: Create natural-sounding RAG context
        if context_parts:
            # Format based on agent personality
            personality_intros = {
                "CEO": "I've been researching the market landscape and found that",
                "CFO": "Looking at the financial benchmarks, I discovered that", 
                "CTO": "From my technical research, I found that",
                "COO": "Based on operational patterns I've studied,"
            }
            
            intro = personality_intros.get(agent_type, "Based on my research,")
            main_insight = context_parts[0]
            
            # Create natural context
            natural_context = f"{intro} {main_insight}. This gives us valuable insights for our analysis."
        else:
            natural_context = ""
        
        return {
            "context": natural_context,
            "sources": sources,
            "relevance_scores": scores,
            "retrieval_success": len(context_parts) > 0
        }
    
    def retrieve_knowledge(self, agent_type: str, query: str, k: int = None) -> List[Document]:
        """Legacy method for backward compatibility"""
        k = k or self.config['agents'].get(agent_type, {}).get('max_retrieval_results', 3)
        docs_and_scores = self.rag_retrieve_and_rank(agent_type, query, k)
        return [doc for doc, score in docs_and_scores]

# Global RAG instance
try:
    rag_knowledge_manager = RAGKnowledgeManager()
    print("✅ RAG Knowledge Manager initialized successfully")
except Exception as e:
    print(f"⚠️ RAG Knowledge Manager initialization failed: {e}")
    rag_knowledge_manager = None



