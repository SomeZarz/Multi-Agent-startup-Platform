#!/usr/bin/env python3
"""
Setup script for initializing knowledge bases on AWS EC2
"""
import os
import sys
from pathlib import Path

# ENHANCED PATH RESOLUTION - Works from any directory
script_dir = Path(__file__).parent.absolute()
sys.path.append(str(script_dir.parent.parent)) # Add src/ to path

from knowledge_system.knowledge_builder import KnowledgeBaseBuilder

def setup_production_knowledge_bases():
    """Setup knowledge bases for production deployment"""
    print("[INFO] Setting up knowledge bases for production...")
    
    # ENHANCED: Set working directory to src for consistent paths
    src_dir = script_dir.parent.parent
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(src_dir))
        print(f"[INFO] Working from: {os.getcwd()}")
        
        # Use paths relative to script location
        base_dir = script_dir.parent
        
        # Ensure all directories exist
        os.makedirs(base_dir / "vector_stores", exist_ok=True)
        os.makedirs(base_dir / "data_sources", exist_ok=True)
        
        # Build knowledge bases
        builder = KnowledgeBaseBuilder()
        knowledge_bases = builder.build_all_knowledge_bases()
        
        # Verify all knowledge bases were created
        required_paths = [
            base_dir / "vector_stores" / "ceo_market_db",
            base_dir / "vector_stores" / "cfo_funding_db", 
            base_dir / "vector_stores" / "cto_tech_db",
            base_dir / "vector_stores" / "coo_operations_db"
        ]
        
        for path in required_paths:
            if path.exists():
                print(f"[SUCCESS] {path} created successfully")
            else:
                print(f"[ERROR] {path} missing!")
                return False
                
        print("[SUCCESS] Production knowledge bases ready!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False
        
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = setup_production_knowledge_bases()
    sys.exit(0 if success else 1)

