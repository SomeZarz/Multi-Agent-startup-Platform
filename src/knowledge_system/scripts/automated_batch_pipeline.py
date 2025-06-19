#!/usr/bin/env python3
import schedule
import time
import sys
import os
from pathlib import Path

# ENHANCED PATH RESOLUTION - Works from any directory
script_dir = Path(__file__).parent.absolute()
sys.path.append(str(script_dir.parent.parent)) # Add src/ to path

from knowledge_system.scripts.data_ingestion import DataSourceIngestion
from knowledge_system.knowledge_builder import KnowledgeBaseBuilder

def daily_knowledge_refresh():
    """Daily batch update of knowledge bases with enhanced working directory management"""
    print("[INFO] Starting daily knowledge base refresh...")
    
    # ENHANCED: Set working directory to src for consistent paths
    src_dir = script_dir.parent.parent
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(src_dir))
        print(f"[INFO] Working from: {os.getcwd()}")
        
        # Step 1: Ingest new data
        ingestion = DataSourceIngestion()
        agent_mapping = {
            "market": "CEO",
            "funding": "CFO",
            "tech": "CTO", 
            "operations": "COO"
        }
        
        total_articles = 0
        for source_type, agent in agent_mapping.items():
            articles = ingestion.save_batch_data(source_type)
            total_articles += articles
            print(f"[DATA] {agent}: {articles} new articles")
        
        # Step 2: Rebuild knowledge bases if new data exists
        if total_articles > 0:
            print("[INFO] Rebuilding knowledge bases with new data...")
            builder = KnowledgeBaseBuilder()
            knowledge_bases = builder.build_all_knowledge_bases()
            if knowledge_bases:
                print(f"[SUCCESS] Knowledge bases updated with {total_articles} new sources")
            else:
                print("[ERROR] Knowledge base rebuild failed")
        else:
            print("[INFO] No new data found, skipping rebuild")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Batch refresh error: {e}")
        return False
        
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)

def setup_batch_schedule():
    """Setup automated batch processing"""
    # Schedule daily at 2 AM (off-peak for AWS costs)
    schedule.every().day.at("02:00").do(daily_knowledge_refresh)
    
    # Optional: Weekly deep refresh
    schedule.every().sunday.at("01:00").do(daily_knowledge_refresh)
    
    print("[INFO] Batch schedule configured:")
    print(" - Daily refresh: 2:00 AM")
    print(" - Weekly deep refresh: Sunday 1:00 AM")
    
    # Run scheduler
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    # Run once immediately for testing
    success = daily_knowledge_refresh()
    
    if success:
        # Then start scheduler
        setup_batch_schedule()
    else:
        print("[ERROR] Initial refresh failed, not starting scheduler")
        sys.exit(1)

