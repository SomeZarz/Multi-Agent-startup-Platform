import os
import re
import functools
import hashlib
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Import agent creation functions
from agents.ceo import create_ceo_agent
from agents.cfo import create_cfo_agent
from agents.cto import create_cto_agent
from agents.coo import create_coo_agent
from agents.supervisor import create_supervisor_chain

# Import RAG knowledge manager with error handling
try:
    from knowledge_system.knowledge_manager import rag_knowledge_manager
    RAG_AVAILABLE = True if rag_knowledge_manager else False
    print("✅ RAG system loaded successfully" if RAG_AVAILABLE else "⚠️ RAG system initialization failed")
except ImportError as e:
    print(f"⚠️ RAG system not available: {e}")
    RAG_AVAILABLE = False
except Exception as e:
    print(f"⚠️ RAG system error: {e}")
    RAG_AVAILABLE = False

# Load environment variables
load_dotenv()

# --- Enhanced Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next: str
    discussion_phase: str
    topics_discussed: List[str]
    pending_questions: List[str]
    message_count: int
    last_speaker: str
    agent_participation: dict
    response_hashes: dict
    agent_call_counts: dict
    conversation_quality: float
    context_summary: str

# --- ENHANCED TAVILY SEARCH CONFIGURATION ---
def create_enhanced_search_tools():
    """Create agent-specific TavilySearch tools with optimized configurations"""
    
    # CEO Tools - Strategic and Market Focus
    ceo_tools = [TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_domains=["crunchbase.com", "techcrunch.com", "forbes.com", "bloomberg.com"],
        exclude_domains=["reddit.com", "quora.com"]
    )]
    
    # CFO Tools - Financial and Funding Focus  
    cfo_tools = [TavilySearch(
        max_results=4,
        search_depth="advanced", 
        include_answer=True,
        include_raw_content=False,
        include_domains=["crunchbase.com", "pitchbook.com", "bloomberg.com", "reuters.com", "sec.gov"],
        exclude_domains=["reddit.com", "quora.com", "wikipedia.org"]
    )]
    
    # CTO Tools - Technical and Development Focus
    cto_tools = [TavilySearch(
        max_results=4,
        search_depth="basic",
        include_answer=True,
        include_raw_content=False,
        include_domains=["github.com", "stackoverflow.com", "medium.com", "dev.to", "techcrunch.com"],
        exclude_domains=["reddit.com", "quora.com"]
    )]
    
    # COO Tools - Operations and Execution Focus
    coo_tools = [TavilySearch(
        max_results=3,
        search_depth="basic",
        include_answer=True,
        include_raw_content=False,
        include_domains=["harvard.edu", "mckinsey.com", "bcg.com", "techcrunch.com", "inc.com"],
        exclude_domains=["reddit.com", "quora.com", "wikipedia.org"]
    )]
    
    return {
        "CEO": ceo_tools,
        "CFO": cfo_tools, 
        "CTO": cto_tools,
        "COO": coo_tools
    }

# --- LLM and Enhanced Tools Creation ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
enhanced_tools = create_enhanced_search_tools()

# Create worker agents with specialized search tools
ceo_agent_executor = create_ceo_agent(llm, enhanced_tools["CEO"])
cfo_agent_executor = create_cfo_agent(llm, enhanced_tools["CFO"])
cto_agent_executor = create_cto_agent(llm, enhanced_tools["CTO"])
coo_agent_executor = create_coo_agent(llm, enhanced_tools["COO"])

# Create the supervisor agent
members = ["CEO", "CTO", "CFO", "COO"]
supervisor_chain = create_supervisor_chain(llm, members)

# --- Helper Functions ---
def extract_business_idea_from_messages(messages):
    """Extract business idea from initial messages"""
    if not messages:
        return ""
    
    initial_message = messages[0].content
    patterns = [
        r"Business Idea:\s*(.+?)(?:\n|$)",
        r"business idea[:\s]+(.+?)(?:\n|$)",
        r"Analyze.*?:\s*(.+?)(?:\n|$)",
        r"consultation[:\s]+(.+?)(?:\n|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, initial_message, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return initial_message[:200].strip()

def calculate_conversation_quality(messages, message_count):
    """Calculate conversation quality degradation"""
    if message_count <= 5:
        return 1.0
    
    recent_messages = messages[-3:] if len(messages) >= 3 else messages
    recent_content = [msg.content for msg in recent_messages if hasattr(msg, 'content')]
    
    avg_length = sum(len(content) for content in recent_content) / len(recent_content) if recent_content else 0
    length_quality = min(avg_length / 500, 1.0)
    count_penalty = max(0.3, 1.0 - (message_count - 5) * 0.1)
    
    return length_quality * count_penalty

def create_context_summary(messages, business_idea):
    """Create rolling summary of conversation"""
    if len(messages) <= 5:
        return f"Analyzing business idea: {business_idea}"
    
    key_points = []
    agent_contributions = {"CEO": [], "CTO": [], "CFO": [], "COO": []}
    
    for msg in messages[-8:]:
        if hasattr(msg, 'name') and msg.name in agent_contributions:
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            agent_contributions[msg.name].append(content)
    
    summary_parts = [f"Business: {business_idea}"]
    for agent, contributions in agent_contributions.items():
        if contributions:
            latest = contributions[-1]
            summary_parts.append(f"{agent}: {latest}")
    
    return " | ".join(summary_parts)

def generate_response_hash(content):
    """Generate hash of response content for deduplication"""
    normalized = re.sub(r'\s+', ' ', content.lower().strip())
    buzzwords = ['innovative', 'cutting-edge', 'scalable', 'disruptive', 'game-changing', 'exciting', 'strategic']
    for word in buzzwords:
        normalized = normalized.replace(word, '')
    
    return hashlib.md5(normalized.encode()).hexdigest()[:10]

def is_response_too_similar(content, previous_hashes, threshold=0.8):
    """Enhanced similarity detection"""
    current_hash = generate_response_hash(content)
    recent_hashes = list(previous_hashes.values())[-5:]
    
    for prev_hash in recent_hashes:
        if current_hash == prev_hash:
            return True
    
    if len(content.lower()) < 100:
        return True
        
    return False

def should_end_conversation(state):
    """Dynamic conversation ending based on quality and completeness"""
    message_count = state.get("message_count", 0)
    agent_participation = state.get("agent_participation", {})
    conversation_quality = state.get("conversation_quality", 1.0)
    
    if conversation_quality < 0.4 and message_count >= 6:
        print("🔄 Ending consultation due to quality degradation")
        return True
    
    if message_count >= 8 and all(agent_participation.values()):
        return True
    
    if message_count >= 12:
        print("🔄 Ending consultation due to message limit")
        return True
    
    return False

def extract_questions_for_others(content, current_agent):
    """Extract questions directed at other agents"""
    questions = []
    agent_patterns = {
        "CEO": ["Sarah", "CEO"],
        "CTO": ["Mike", "CTO"], 
        "CFO": ["Jennifer", "CFO"],
        "COO": ["Tom", "COO"]
    }
    
    for agent, names in agent_patterns.items():
        if agent != current_agent:
            for name in names:
                pattern = rf"{name},\s*([^?]*\?)"
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    questions.append(f"{agent}: {name}, {match}")
    return questions

def extract_topics_discussed(content):
    """Extract key topics from agent message"""
    topics = []
    topic_keywords = {
        "technical": ["technology", "tech stack", "architecture", "development", "MVP", "scalability"],
        "financial": ["budget", "funding", "revenue", "costs", "financial", "money", "valuation", "burn rate"],
        "market": ["market", "competition", "customers", "users", "TAM", "SAM", "competitive analysis"],
        "operations": ["operations", "hiring", "timeline", "execution", "team", "go-to-market"],
        "strategy": ["strategy", "vision", "goals", "planning", "roadmap", "business model"]
    }
    
    content_lower = content.lower()
    for topic, keywords in topic_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            topics.append(topic)
    return topics

def check_for_final_report(content):
    """Enhanced final report detection"""
    content_upper = content.upper()
    final_report_indicators = [
        "FINAL REPORT:",
        "FINAL REPORT ",
        "EXECUTIVE SUMMARY",
        "## EXECUTIVE SUMMARY", 
        "**FINAL RECOMMENDATION**",
        "FINAL RECOMMENDATION:"
    ]
    
    has_indicator = any(indicator in content_upper for indicator in final_report_indicators)
    is_substantial = len(content) > 500
    return has_indicator and is_substantial

# --- Enhanced Worker Node Function with Full RAG Integration ---
def worker_node(state, agent, name):
    """Enhanced worker node with full RAG implementation and personality preservation"""
    last_speaker = state.get("last_speaker", "")
    message_count = state.get("message_count", 0)
    agent_participation = state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False})
    response_hashes = state.get("response_hashes", {})
    agent_call_counts = state.get("agent_call_counts", {"CEO": 0, "CTO": 0, "CFO": 0, "COO": 0})
    conversation_quality = state.get("conversation_quality", 1.0)
    context_summary = state.get("context_summary", "")
    
    # Check if should end conversation early
    if should_end_conversation(state) and name == "CEO":
        is_final_report_time = True
    else:
        is_final_report_time = (
            message_count >= 8 and
            all(agent_participation.values()) and
            name == "CEO"
        )
    
    # Apply "no consecutive speaker" rule EXCEPT for final report
    if last_speaker == name and not is_final_report_time:
        return {
            "messages": [],
            "next": "supervisor",
            "skip_reason": f"{name}_consecutive_speaker_prevention"
        }
    
    # Track agent call frequency
    current_call_count = agent_call_counts.get(name, 0) + 1
    
    # Prepare modified state for agent processing
    modified_state = dict(state)
    additional_messages = []
    
    # Add context summary for focus
    if context_summary and current_call_count > 1:
        summary_message = HumanMessage(
            content=f"Conversation context: {context_summary}",
            name="context_summary"
        )
        additional_messages.append(summary_message)
    
    # FULL RAG IMPLEMENTATION - Natural Integration
    if RAG_AVAILABLE and current_call_count <= 2:
        try:
            business_context = extract_business_idea_from_messages(state.get("messages", []))
            
            if business_context and len(business_context) > 10:
                # Generate RAG context
                rag_result = rag_knowledge_manager.rag_generate_context(
                    name, business_context, "business analysis consultation"
                )
                
                if rag_result["retrieval_success"] and rag_result["context"]:
                    # Add RAG context as natural background information
                    rag_message = HumanMessage(
                        content=rag_result["context"],
                        name=f"{name.lower()}_background_research"
                    )
                    additional_messages.append(rag_message)
                    
                    print(f"🔍 RAG enhanced {name} with {len(rag_result['sources'])} knowledge sources")
                
        except Exception as e:
            print(f"⚠️ RAG error for {name}: {e}")
    
    # Add personality reinforcement for later calls
    if current_call_count > 2:
        personality_reinforcement = {
            "CEO": "Stay enthusiastic and visionary. Use 'You know what?' and build excitement.",
            "CFO": "Keep using relatable analogies. Stay practical and numbers-focused.", 
            "CTO": "Be straightforward and honest about technical realities.",
            "COO": "Focus on practical execution. Bridge vision to reality."
        }
        
        personality_message = HumanMessage(
            content=f"Personality reminder: {personality_reinforcement.get(name, 'Stay authentic to your role.')} Avoid being robotic or overly formal.",
            name="personality_context"
        )
        additional_messages.append(personality_message)
    
    # Add final report instruction if needed
    if is_final_report_time:
        final_report_message = HumanMessage(
            content="The consultation is ready for conclusion. Please provide your comprehensive final report with strategic recommendations.",
            name="system"
        )
        additional_messages.append(final_report_message)
    
    # Update modified state with additional messages
    if additional_messages:
        modified_state["messages"] = state["messages"] + additional_messages
        result = agent.invoke(modified_state)
    else:
        result = agent.invoke(state)
    
    content = result["output"]
    
    # Enhanced repetition detection for end-stage
    if is_response_too_similar(content, response_hashes, threshold=0.7):
        print(f"⚠️ {name} generated similar response, regenerating...")
        
        variety_message = HumanMessage(
            content=f"Your response was too similar to previous messages. Please provide a fresh, distinct perspective. Focus on a completely different aspect or provide new insights.",
            name="anti_repetition"
        )
        modified_state["messages"] = modified_state["messages"] + [variety_message]
        result = agent.invoke(modified_state)
        content = result["output"]
    
    # Generate hash for current response
    current_hash = generate_response_hash(content)
    response_hashes[f"{name}_{message_count}"] = current_hash
    
    # Extract questions and topics
    questions = extract_questions_for_others(content, name)
    new_topics = extract_topics_discussed(content)
    existing_topics = state.get("topics_discussed", [])
    combined_topics = list(set(existing_topics + new_topics))
    
    # Update participation and counts
    updated_participation = dict(agent_participation)
    updated_participation[name] = True
    
    updated_call_counts = dict(agent_call_counts)
    updated_call_counts[name] = current_call_count
    
    # Increment message count
    new_count = message_count + 1
    
    # Calculate new conversation quality
    all_messages = state.get("messages", []) + [HumanMessage(content=content, name=name)]
    new_quality = calculate_conversation_quality(all_messages, new_count)
    
    # Update context summary
    business_idea = extract_business_idea_from_messages(state.get("messages", []))
    new_summary = create_context_summary(all_messages, business_idea)
    
    # Determine discussion phase
    if new_count <= 4:
        phase = "initial"
    elif new_count <= 8:
        phase = "discussion"
    else:
        phase = "synthesis"
    
    return {
        "messages": [HumanMessage(content=content, name=name)],
        "pending_questions": questions,
        "topics_discussed": combined_topics,
        "message_count": new_count,
        "discussion_phase": phase,
        "last_speaker": name,
        "agent_participation": updated_participation,
        "response_hashes": response_hashes,
        "agent_call_counts": updated_call_counts,
        "conversation_quality": new_quality,
        "context_summary": new_summary
    }

# Define nodes for each worker agent
ceo_node = functools.partial(worker_node, agent=ceo_agent_executor, name="CEO")
cto_node = functools.partial(worker_node, agent=cto_agent_executor, name="CTO") 
cfo_node = functools.partial(worker_node, agent=cfo_agent_executor, name="CFO")
coo_node = functools.partial(worker_node, agent=coo_agent_executor, name="COO")

# --- Enhanced Supervisor Node ---
def supervisor_node(state):
    """Enhanced supervisor with quality-aware routing"""
    messages = state.get("messages", [])
    message_count = state.get("message_count", 0)
    last_speaker = state.get("last_speaker", "")
    agent_participation = state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False})
    pending_questions = state.get("pending_questions", [])
    topics_discussed = state.get("topics_discussed", [])
    agent_call_counts = state.get("agent_call_counts", {"CEO": 0, "CTO": 0, "CFO": 0, "COO": 0})
    conversation_quality = state.get("conversation_quality", 1.0)
    
    # Check for skip reason
    if state.get("skip_reason"):
        print(f"🔄 Routing skipped due to: {state['skip_reason']}")
    
    # Check if we already have a final report
    if messages:
        last_message = messages[-1].content
        if check_for_final_report(last_message):
            return {"next": "FINISH"}
    
    # DYNAMIC ENDING: End early if quality degraded or natural completion
    if should_end_conversation(state):
        return {"next": "CEO"}
    
    # RULE 1: Ensure each agent participates at least once
    for agent in ["CEO", "CTO", "CFO", "COO"]:
        if not agent_participation[agent] and agent != last_speaker:
            return {"next": agent}
    
    # RULE 2: Handle pending questions (but not to last speaker)
    if pending_questions and message_count < 8:
        latest_question = pending_questions[-1]
        target_agent = None
        if "CEO:" in latest_question:
            target_agent = "CEO"
        elif "CTO:" in latest_question:
            target_agent = "CTO"
        elif "CFO:" in latest_question:
            target_agent = "CFO"
        elif "COO:" in latest_question:
            target_agent = "COO"
        
        if target_agent and target_agent != last_speaker:
            return {"next": target_agent}
    
    # RULE 3: Topic-based routing
    if topics_discussed and message_count < 8:
        latest_topics = topics_discussed[-2:] if len(topics_discussed) >= 2 else topics_discussed
        
        if "technical" in latest_topics and last_speaker != "CTO":
            return {"next": "CTO"}
        elif "financial" in latest_topics and last_speaker != "CFO":
            return {"next": "CFO"}
        elif "operations" in latest_topics and last_speaker != "COO":
            return {"next": "COO"}
        elif "strategy" in latest_topics and last_speaker != "CEO":
            return {"next": "CEO"}
    
    # RULE 4: Route to least active agent (excluding last speaker)
    if message_count < 8:
        available_agents = [agent for agent in ["CEO", "CTO", "CFO", "COO"] if agent != last_speaker]
        if available_agents:
            least_active = min(available_agents, key=lambda x: agent_call_counts.get(x, 0))
            return {"next": least_active}
    
    # RULE 5: Force conclusion
    return {"next": "CEO"}

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("CEO", ceo_node)
workflow.add_node("CTO", cto_node)
workflow.add_node("CFO", cfo_node)
workflow.add_node("COO", coo_node)
workflow.add_node("supervisor", supervisor_node)

# Add edges from each worker back to the supervisor
for member in members:
    workflow.add_edge(member, "supervisor")

# Conditional routing
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

# Compile the graph
app = workflow.compile()

def main():
    idea = input("Please enter your business idea: ")
    initial_messages = [HumanMessage(content=f"Analyze the following business idea and provide comprehensive consultation with data-driven insights. Business Idea: {idea}")]
    
    initial_state = {
        "messages": initial_messages,
        "discussion_phase": "initial",
        "topics_discussed": [],
        "pending_questions": [],
        "message_count": 0,
        "last_speaker": "",
        "agent_participation": {"CEO": False, "CTO": False, "CFO": False, "COO": False},
        "response_hashes": {},
        "agent_call_counts": {"CEO": 0, "CTO": 0, "CFO": 0, "COO": 0},
        "conversation_quality": 1.0,
        "context_summary": ""
    }
    
    print("\n--- Starting RAG-Enhanced AI Startup Consultation ---")
    if RAG_AVAILABLE:
        print("🔍 Full RAG system active: Agents have access to knowledge bases and real-time market data")
    else:
        print("🔍 Agents have access to real-time market data via TavilySearch")
    print("🎯 Dynamic quality management and anti-repetition system active\n")
    
    # Add recursion limit to prevent infinite loops
    config = {"recursion_limit": 30}
    
    for output in app.stream(initial_state, config=config):
        for key, value in output.items():
            if key != "__end__" and "messages" in value and value['messages']:
                agent_name = key.upper()
                agent_message = value['messages'][-1].content
                
                # Show quality indicator
                quality = value.get('conversation_quality', 1.0)
                quality_indicator = "🟢" if quality > 0.7 else "🟡" if quality > 0.4 else "🔴"
                
                print(f"--- {agent_name} {quality_indicator} ---")
                print(agent_message)
                print()
    
    print("--- RAG-Enhanced Consultation Finished ---")
    if RAG_AVAILABLE:
        print("💡 Your consultation included RAG-powered knowledge insights and real-time market research!")
    else:
        print("💡 Your consultation included real-time market research!")
    print("🎯 Conversation optimized for quality and personality preservation!")

if __name__ == "__main__":
    main()

