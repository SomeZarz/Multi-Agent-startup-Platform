# src/knowledge_system/scripts/data_ingestion.py
import requests
import json
import feedparser
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys
import os

# ENHANCED PATH RESOLUTION - Works from any directory
script_dir = Path(__file__).parent.absolute()
sys.path.append(str(script_dir.parent.parent)) # Add src/ to path

from knowledge_system.knowledge_builder import KnowledgeBaseBuilder

class DataSourceIngestion:
    def __init__(self):
        # Use portable paths relative to script location
        self.base_path = script_dir.parent / "data_sources"
        
        # Agent-specific data mapping
        self.agent_sources = {
            "market": {
                "feeds": [
                    "https://feeds.bloomberg.com/startup-news.rss",
                    "https://techcrunch.com/startups/feed/",
                    "https://feeds.crunchbase.com/news.rss"
                ]
            },
            "funding": {
                "feeds": [
                    "https://feeds.crunchbase.com/funding.rss",
                    "https://venturebeat.com/business/feed/"
                ]
            },
            "tech": {
                "feeds": [
                    "https://stackoverflow.blog/feed/",
                    "https://github.blog/feed/"
                ]
            },
            "operations": {
                "feeds": [
                    "https://hbr.org/feed"
                ]
            }
        }

    def fetch_rss_content(self, agent_type: str, max_articles: int = 20):
        """Fetch RSS content for specific agent type"""
        articles = []
        feeds = self.agent_sources.get(agent_type, {}).get("feeds", [])
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_articles]:
                    articles.append({
                        "title": entry.title,
                        "content": getattr(entry, 'summary', ''),
                        "published": getattr(entry, 'published', ''),
                        "source": feed_url
                    })
                print(f"[SUCCESS] Fetched {len(feed.entries[:max_articles])} articles from {feed_url}")
            except Exception as e:
                print(f"[ERROR] Failed to fetch {feed_url}: {e}")
        
        return articles

    def save_batch_data(self, agent_type: str):
        """Save fetched data for knowledge base ingestion"""
        agent_dir = self.base_path / f"{agent_type}_data"
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch content
        articles = self.fetch_rss_content(agent_type)
        
        if articles:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Save as text for knowledge base (easier processing)
            txt_file = agent_dir / f"{agent_type}_insights_{timestamp}.txt"
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                for article in articles:
                    f.write(f"Title: {article['title']}\n")
                    f.write(f"Content: {article['content']}\n")
                    f.write(f"Published: {article['published']}\n")
                    f.write("-" * 80 + "\n")
            
            print(f"[SUCCESS] Saved {len(articles)} articles for {agent_type}")
            return len(articles)
        
        return 0

def main():
    """Main data ingestion function with enhanced working directory management"""
    print("[INFO] Starting data ingestion...")
    
    # ENHANCED: Set working directory to src for consistent paths
    src_dir = script_dir.parent.parent
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(src_dir))
        print(f"[INFO] Working from: {os.getcwd()}")
        
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
        
        print(f"[SUCCESS] Data ingestion completed! Total articles: {total_articles}")
        
        # Rebuild knowledge bases if new data exists
        if total_articles > 0:
            print("[INFO] Rebuilding knowledge bases with new data...")
            try:
                builder = KnowledgeBaseBuilder()
                knowledge_bases = builder.build_all_knowledge_bases()
                if knowledge_bases:
                    print(f"[SUCCESS] Knowledge bases updated with {total_articles} new sources")
                else:
                    print("[ERROR] Knowledge base rebuild failed")
            except Exception as e:
                print(f"[ERROR] Error rebuilding knowledge bases: {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Data ingestion error: {e}")
        return False
        
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
