import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Add parent directories to path for imports  
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent.parent))  # Add src/ to path

class RealtimeFeedMonitor:
    def __init__(self):
        # Use portable paths relative to script location
        self.base_path = script_dir.parent / "data_sources"
        self.last_fetch_times = {}
        
        # High-priority feeds for real-time monitoring
        self.priority_feeds = {
            "market": [
                "https://techcrunch.com/startups/feed/",
                "https://feeds.bloomberg.com/startup-news.rss"
            ],
            "funding": [
                "https://feeds.crunchbase.com/funding.rss"
            ],
            "tech": [
                "https://stackoverflow.blog/feed/"
            ],
            "operations": [
                "https://hbr.org/feed"
            ]
        }
    
    async def monitor_feed(self, agent_type: str, feed_url: str):
        """Monitor single feed for new content"""
        try:
            # Get last fetch time
            last_fetch = self.last_fetch_times.get(f"{agent_type}_{feed_url}", 
                                                  datetime.now() - timedelta(hours=6))
            
            feed = feedparser.parse(feed_url)
            new_articles = []
            
            for entry in feed.entries:
                # Check if article is newer than last fetch
                entry_time = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                
                if entry_time > last_fetch:
                    new_articles.append({
                        "title": entry.title,
                        "content": getattr(entry, 'summary', ''),
                        "published": str(entry_time),
                        "source": feed_url,
                        "urgent": True  # Mark as real-time update
                    })
            
            if new_articles:
                await self.save_realtime_update(agent_type, new_articles)
                self.last_fetch_times[f"{agent_type}_{feed_url}"] = datetime.now()
                print(f"🔴 REAL-TIME: {len(new_articles)} new articles for {agent_type}")
            
        except Exception as e:
            print(f"❌ Real-time monitor error for {feed_url}: {e}")
    
    async def save_realtime_update(self, agent_type: str, articles: list):
        """Save real-time updates immediately"""
        agent_dir = self.base_path / f"{agent_type}_data"
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as urgent update
        urgent_file = agent_dir / f"{agent_type}_urgent_{timestamp}.txt"
        with open(urgent_file, 'w', encoding='utf-8') as f:
            f.write("URGENT MARKET UPDATE\n")
            f.write("=" * 50 + "\n")
            for article in articles:
                f.write(f"BREAKING: {article['title']}\n")
                f.write(f"Content: {article['content']}\n")
                f.write(f"Published: {article['published']}\n")
                f.write("-" * 30 + "\n")
        
        # Trigger immediate knowledge base update for urgent news
        if len(articles) > 0:
            await self.trigger_urgent_rebuild(agent_type)
    
    async def trigger_urgent_rebuild(self, agent_type: str):
        """Trigger immediate knowledge base rebuild for urgent updates"""
        try:
            from knowledge_system.knowledge_builder import KnowledgeBaseBuilder
            builder = KnowledgeBaseBuilder()
            
            # Rebuild specific agent's knowledge base
            if agent_type == "market":
                builder.build_ceo_knowledge_base()
            elif agent_type == "funding":
                builder.build_cfo_knowledge_base()
            elif agent_type == "tech":
                builder.build_cto_knowledge_base()
            elif agent_type == "operations":
                builder.build_coo_knowledge_base()
            
            print(f"🚨 URGENT: {agent_type} knowledge base updated immediately")
        except Exception as e:
            print(f"❌ Urgent rebuild failed: {e}")
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        print("🔴 Starting real-time feed monitoring...")
        
        while True:
            tasks = []
            for agent_type, feeds in self.priority_feeds.items():
                for feed_url in feeds:
                    tasks.append(self.monitor_feed(agent_type, feed_url))
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(1800)  # Check every 30 minutes

# Run real-time monitor
if __name__ == "__main__":
    monitor = RealtimeFeedMonitor()
    asyncio.run(monitor.start_monitoring())


