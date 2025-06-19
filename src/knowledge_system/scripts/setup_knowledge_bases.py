#!/usr/bin/env python3
"""
Setup script for initializing knowledge bases on AWS EC2
"""

import os
import sys
sys.path.append('..')

from knowledge_builder import KnowledgeBaseBuilder

def setup_production_knowledge_bases():
    """Setup knowledge bases for production deployment"""
    
    print("🚀 Setting up knowledge bases for production...")
    
    # Ensure all directories exist
    os.makedirs("knowledge_system/vector_stores", exist_ok=True)
    os.makedirs("knowledge_system/data_sources", exist_ok=True)
    
    # Build knowledge bases
    builder = KnowledgeBaseBuilder()
    knowledge_bases = builder.build_all_knowledge_bases()
    
    # Verify all knowledge bases were created
    required_paths = [
        "knowledge_system/vector_stores/ceo_market_db",
        "knowledge_system/vector_stores/cfo_funding_db", 
        "knowledge_system/vector_stores/cto_tech_db",
        "knowledge_system/vector_stores/coo_operations_db"
    ]
    
    for path in required_paths:
        if os.path.exists(path):
            print(f"✅ {path} created successfully")
        else:
            print(f"❌ {path} missing!")
            return False
    
    print("🎉 Production knowledge bases ready!")
    return True

if __name__ == "__main__":
    success = setup_production_knowledge_bases()
    sys.exit(0 if success else 1)
