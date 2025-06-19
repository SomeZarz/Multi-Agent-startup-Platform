import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv  # Add this import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, JSONLoader, CSVLoader  # Updated imports
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

class KnowledgeBaseBuilder:
    def __init__(self, config_path="knowledge_system/config/kb_config.yaml"):
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please ensure OPENAI_API_KEY is set in your .env file or environment variables."
            )
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file not found at {config_path}. Using default configuration.")
            self.config = self._get_default_config()
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Cost-effective for your GPT-3.5-turbo setup
            openai_api_key=os.getenv("OPENAI_API_KEY")  # Explicitly pass the API key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.base_path = Path("knowledge_system")
    
    def _get_default_config(self):
        """Default configuration if config file is missing"""
        return {
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "vector_store_type": "FAISS",
            "agents": {
                "CEO": {
                    "domains": ["market_analysis", "strategy"],
                    "max_retrieval_results": 5,
                    "relevance_threshold": 0.7
                },
                "CFO": {
                    "domains": ["financial_metrics", "funding"],
                    "max_retrieval_results": 4,
                    "relevance_threshold": 0.75
                },
                "CTO": {
                    "domains": ["technology", "development"],
                    "max_retrieval_results": 4,
                    "relevance_threshold": 0.7
                },
                "COO": {
                    "domains": ["operations", "strategy"],
                    "max_retrieval_results": 3,
                    "relevance_threshold": 0.7
                }
            }
        }
    
    def build_ceo_knowledge_base(self):
        """Build CEO market intelligence knowledge base"""
        print("🏢 Building CEO Market Intelligence Knowledge Base...")
        
        documents = []
        
        # Load market intelligence data
        market_data_path = self.base_path / "data_sources" / "market_data"
        documents.extend(self._load_documents_from_directory(market_data_path))
        
        # Add startup ecosystem insights
        startup_insights = self._get_startup_ecosystem_data()
        documents.extend(startup_insights)
        
        # Build and save vector store
        vector_store = self._create_vector_store(documents)
        vector_store.save_local("knowledge_system/vector_stores/ceo_market_db")
        
        print(f"✅ CEO Knowledge Base created with {len(documents)} documents")
        return vector_store
    
    def build_cfo_knowledge_base(self):
        """Build CFO funding and financial knowledge base"""
        print("💰 Building CFO Financial Knowledge Base...")
        
        documents = []
        
        # Load funding data
        funding_data_path = self.base_path / "data_sources" / "funding_data"
        documents.extend(self._load_documents_from_directory(funding_data_path))
        
        # Add financial benchmarks
        financial_data = self._get_financial_benchmarks()
        documents.extend(financial_data)
        
        # Build and save vector store
        vector_store = self._create_vector_store(documents)
        vector_store.save_local("knowledge_system/vector_stores/cfo_funding_db")
        
        print(f"✅ CFO Knowledge Base created with {len(documents)} documents")
        return vector_store
    
    def build_cto_knowledge_base(self):
        """Build CTO technology trends knowledge base"""
        print("💻 Building CTO Technology Knowledge Base...")
        
        documents = []
        
        # Load tech data
        tech_data_path = self.base_path / "data_sources" / "tech_data"
        documents.extend(self._load_documents_from_directory(tech_data_path))
        
        # Add technology trends
        tech_trends = self._get_technology_trends()
        documents.extend(tech_trends)
        
        # Build and save vector store
        vector_store = self._create_vector_store(documents)
        vector_store.save_local("knowledge_system/vector_stores/cto_tech_db")
        
        print(f"✅ CTO Knowledge Base created with {len(documents)} documents")
        return vector_store
    
    def build_coo_knowledge_base(self):
        """Build COO operations knowledge base"""
        print("⚙️ Building COO Operations Knowledge Base...")
        
        documents = []
        
        # Load operations data
        ops_data_path = self.base_path / "data_sources" / "operations_data"
        documents.extend(self._load_documents_from_directory(ops_data_path))
        
        # Add operational best practices
        ops_data = self._get_operational_best_practices()
        documents.extend(ops_data)
        
        # Build and save vector store
        vector_store = self._create_vector_store(documents)
        vector_store.save_local("knowledge_system/vector_stores/coo_operations_db")
        
        print(f"✅ COO Knowledge Base created with {len(documents)} documents")
        return vector_store
    
    def _load_documents_from_directory(self, directory_path: Path) -> List[Document]:
        """Load documents from various file formats"""
        documents = []
        
        if not directory_path.exists():
            print(f"📁 Directory {directory_path} doesn't exist, skipping...")
            return documents
        
        for file_path in directory_path.glob("**/*"):
            if file_path.is_file():
                try:
                    if file_path.suffix == '.txt':
                        loader = TextLoader(str(file_path))
                        documents.extend(loader.load())
                    elif file_path.suffix == '.json':
                        loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
                        documents.extend(loader.load())
                    elif file_path.suffix == '.csv':
                        loader = CSVLoader(str(file_path))
                        documents.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Error loading {file_path}: {e}")
        
        return documents
    
    def _get_startup_ecosystem_data(self) -> List[Document]:
        """Curated startup ecosystem insights for CEO"""
        startup_data = [
            {
                "content": """Startup Market Trends 2025:
                - SaaS startups average $2.3M Series A funding
                - Fintech startups show 35% YoY growth
                - AI/ML startups comprise 23% of new ventures
                - Average time to Series A: 18 months
                - Customer acquisition costs increased 15% in 2024""",
                "metadata": {"source": "startup_trends_2025", "domain": "market_analysis"}
            },
            {
                "content": """Successful Startup Patterns:
                - Product-market fit indicators: 40%+ organic growth, <5% churn
                - MVP timeline: 3-6 months typical, 6-12 months for complex products
                - Team composition: 2-3 founders optimal, technical + business mix
                - Revenue models: Subscription (65%), Marketplace (20%), Enterprise (15%)""",
                "metadata": {"source": "startup_patterns", "domain": "strategy"}
            }
        ]
        
        return [Document(page_content=item["content"], metadata=item["metadata"]) for item in startup_data]
    
    def _get_financial_benchmarks(self) -> List[Document]:
        """Financial benchmarks for CFO"""
        financial_data = [
            {
                "content": """SaaS Financial Benchmarks 2025:
                - Median burn rate: $150K/month for Series A
                - Revenue multiples: 8-12x for growth companies
                - Customer LTV/CAC ratio: 3:1 minimum, 5:1 excellent
                - Gross margins: 70-80% for SaaS, 40-60% for marketplace
                - Runway recommendations: 18-24 months post-funding""",
                "metadata": {"source": "saas_benchmarks", "domain": "financial_metrics"}
            },
            {
                "content": """Startup Funding Landscape 2025:
                - Pre-seed: $250K-$1M, 6-18 months runway
                - Seed: $1M-$5M, 12-24 months runway  
                - Series A: $5M-$15M, 24-36 months runway
                - Valuation trends: 10-15x revenue for profitable SaaS""",
                "metadata": {"source": "funding_landscape", "domain": "funding"}
            }
        ]
        
        return [Document(page_content=item["content"], metadata=item["metadata"]) for item in financial_data]
    
    def _get_technology_trends(self) -> List[Document]:
        """Technology trends for CTO"""
        tech_data = [
            {
                "content": """Technology Stack Trends 2025:
                - Frontend: React (45%), Vue.js (25%), Angular (15%)
                - Backend: Node.js (40%), Python (35%), Go (15%)
                - Database: PostgreSQL (50%), MongoDB (25%), Redis (20%)
                - Cloud: AWS (45%), Azure (25%), GCP (20%)
                - DevOps: Docker (80%), Kubernetes (60%), CI/CD (90%)""",
                "metadata": {"source": "tech_trends_2025", "domain": "technology"}
            },
            {
                "content": """MVP Development Guidelines:
                - Timeline: 3-6 months for web apps, 6-12 months for mobile
                - Team size: 2-4 developers optimal for MVP
                - Architecture: Microservices for scalability, monolith for speed
                - Testing: 20% of development time, automated testing essential""",
                "metadata": {"source": "mvp_guidelines", "domain": "development"}
            }
        ]
        
        return [Document(page_content=item["content"], metadata=item["metadata"]) for item in tech_data]
    
    def _get_operational_best_practices(self) -> List[Document]:
        """Operational best practices for COO"""
        ops_data = [
            {
                "content": """Startup Hiring Best Practices:
                - Technical roles: 4-8 weeks hiring cycle
                - Non-technical roles: 2-6 weeks hiring cycle
                - Contractor vs FTE: Use contractors for first 6 months
                - Equity distribution: 10-20% for first 10 employees
                - Remote work: 70% of startups now remote-first""",
                "metadata": {"source": "hiring_practices", "domain": "operations"}
            },
            {
                "content": """Go-to-Market Strategy Framework:
                - Customer discovery: 50-100 interviews minimum
                - Beta testing: 10-25 beta customers for B2B
                - Launch timeline: 3-6 months marketing runway
                - Pricing strategy: Start high, iterate based on feedback
                - Sales process: Inside sales for <$10K ACV, field sales for >$50K ACV""",
                "metadata": {"source": "gtm_framework", "domain": "strategy"}
            }
        ]
        
        return [Document(page_content=item["content"], metadata=item["metadata"]) for item in ops_data]
    
    def _create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        if not documents:
            # Create empty vector store with dummy document
            documents = [Document(page_content="No data available", metadata={"source": "empty"})]
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(texts, self.embeddings)
        
        return vector_store
    
    def build_all_knowledge_bases(self):
        """Build all knowledge bases"""
        print("🚀 Starting Knowledge Base Construction...")
        print(f"📍 Working directory: {os.getcwd()}")
        print(f"🔑 OpenAI API Key: {'✅ Found' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
        
        # Ensure directories exist
        os.makedirs("knowledge_system/vector_stores", exist_ok=True)
        
        # Build each knowledge base
        try:
            ceo_kb = self.build_ceo_knowledge_base()
            cfo_kb = self.build_cfo_knowledge_base()
            cto_kb = self.build_cto_knowledge_base()
            coo_kb = self.build_coo_knowledge_base()
            
            print("🎉 All Knowledge Bases Built Successfully!")
            
            return {
                "CEO": ceo_kb,
                "CFO": cfo_kb,
                "CTO": cto_kb,
                "COO": coo_kb
            }
        except Exception as e:
            print(f"❌ Error building knowledge bases: {e}")
            return None

# Usage
if __name__ == "__main__":
    try:
        builder = KnowledgeBaseBuilder()
        knowledge_bases = builder.build_all_knowledge_bases()
        if knowledge_bases:
            print("\n✅ Knowledge Base Setup Complete!")
            print("You can now proceed to test the knowledge manager.")
        else:
            print("\n❌ Knowledge Base Setup Failed!")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure your .env file contains OPENAI_API_KEY=your_key_here")
        print("2. Verify you have the required packages installed:")
        print("   pip install langchain-openai langchain-community")
