from knowledge_builder import KnowledgeBaseBuilder
from knowledge_manager import KnowledgeManager

def test_knowledge_system():
    """Test the complete knowledge system"""
    
    # Step 1: Build knowledge bases
    print("🔨 Building knowledge bases...")
    builder = KnowledgeBaseBuilder()
    knowledge_bases = builder.build_all_knowledge_bases()
    
    # Step 2: Test knowledge manager
    print("\n🧪 Testing knowledge manager...")
    manager = KnowledgeManager()
    
    # Test queries for each agent
    test_queries = {
        "CEO": "fintech startup market opportunity",
        "CFO": "SaaS funding requirements Series A",
        "CTO": "MVP development timeline technology stack",
        "COO": "startup hiring and go-to-market strategy"
    }
    
    for agent, query in test_queries.items():
        print(f"\n--- Testing {agent} Agent ---")
        knowledge = manager.get_contextual_knowledge(agent, "fintech startup", query)
        print(f"Knowledge Retrieved: {len(knowledge)} characters")
        print(f"Preview: {knowledge[:200]}...")
    
    print("\n✅ Knowledge system test completed!")

if __name__ == "__main__":
    test_knowledge_system()
