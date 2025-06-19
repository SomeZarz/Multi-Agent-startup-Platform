from knowledge_builder import KnowledgeBaseBuilder
from knowledge_manager import RAGKnowledgeManager

def test_knowledge_system():
    """Test the complete knowledge system"""
    
    # Step 1: Build knowledge bases
    print("🔨 Building knowledge bases...")
    builder = KnowledgeBaseBuilder()
    knowledge_bases = builder.build_all_knowledge_bases()
    
    # Step 2: Test knowledge manager
    print("\n🧪 Testing knowledge manager...")
    manager = RAGKnowledgeManager()
    
    # Test queries for each agent
    test_queries = {
        "CEO": "fintech startup market opportunity",
        "CFO": "SaaS funding requirements Series A", 
        "CTO": "MVP development timeline technology stack",
        "COO": "startup hiring and go-to-market strategy"
    }
    
    for agent, query in test_queries.items():
        print(f"\n--- Testing {agent} Agent ---")
        
        # Use the correct RAG method
        rag_result = manager.rag_generate_context(agent, "fintech startup", query)
        
        if rag_result["retrieval_success"]:
            context = rag_result["context"]
            sources = rag_result["sources"]
            scores = rag_result["relevance_scores"]
            
            print(f"✅ Knowledge Retrieved: {len(context)} characters")
            print(f"📚 Sources: {sources}")
            print(f"📊 Relevance Scores: {scores}")
            print(f"🔍 Preview: {context[:200]}...")
        else:
            print("❌ No relevant knowledge retrieved")
    
    print("\n✅ Knowledge system test completed!")

if __name__ == "__main__":
    test_knowledge_system()

