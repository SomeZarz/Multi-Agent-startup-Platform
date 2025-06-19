import os
import re
import functools
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
    agent_participation: dict  # Track which agents have spoken

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
        search_depth="basic",  # Faster for tech queries
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
    
    # Check for indicators and ensure substantial content
    has_indicator = any(indicator in content_upper for indicator in final_report_indicators)
    is_substantial = len(content) > 500  # Ensure it's not just a brief message
    return has_indicator and is_substantial

def optimize_search_query(query, agent_type):
    """Optimize search queries based on agent expertise"""
    query_optimizations = {
        "CEO": {
            "prefixes": ["startup market trends", "industry analysis", "business strategy"],
            "suffixes": ["market size", "competitive landscape", "industry report 2025"]
        },
        "CFO": {
            "prefixes": ["startup funding", "financial projections", "revenue model"],
            "suffixes": ["funding rounds", "valuation", "financial metrics", "investment trends 2025"]
        },
        "CTO": {
            "prefixes": ["technology stack", "software architecture", "development timeline"],
            "suffixes": ["technical implementation", "scalability", "tech trends 2025"]
        },
        "COO": {
            "prefixes": ["startup operations", "business execution", "go-to-market strategy"],
            "suffixes": ["operational best practices", "execution plan", "team building"]
        }
    }
    
    if agent_type in query_optimizations:
        opts = query_optimizations[agent_type]
        # Keep original query but add context for better results
        return f"{query} {opts['suffixes'][0]}"
    return query

# --- Enhanced Worker Node Function ---
def worker_node(state, agent, name):
    """Enhanced worker node with optimized search capabilities"""
    last_speaker = state.get("last_speaker", "")
    message_count = state.get("message_count", 0)
    agent_participation = state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False})
    
    # CRITICAL FIX: Create exception for final report
    is_final_report_time = (
        message_count >= 10 and
        all(agent_participation.values()) and
        name == "CEO"
    )
    
    # Apply "no consecutive speaker" rule EXCEPT for final report
    if last_speaker == name and not is_final_report_time:
        return state
    
    # Check if should give final report
    if is_final_report_time:
        modified_state = dict(state)
        modified_state["messages"] = state["messages"] + [
            HumanMessage(content="SYSTEM: Please provide the FINAL REPORT summarizing our comprehensive analysis with data-driven insights from your research.", name="system")
        ]
        result = agent.invoke(modified_state)
    else:
        # Normal processing with enhanced context
        result = agent.invoke(state)
    
    content = result["output"]
    
    # Extract questions directed at other agents
    questions = extract_questions_for_others(content, name)
    
    # Extract topics discussed  
    new_topics = extract_topics_discussed(content)
    existing_topics = state.get("topics_discussed", [])
    combined_topics = list(set(existing_topics + new_topics))
    
    # Update agent participation
    updated_participation = dict(state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False}))
    updated_participation[name] = True
    
    # Increment message count
    new_count = state.get("message_count", 0) + 1
    
    # Determine discussion phase
    if new_count <= 4:
        phase = "initial"
    elif new_count <= 10:
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
        "agent_participation": updated_participation
    }

# Define nodes for each worker agent
ceo_node = functools.partial(worker_node, agent=ceo_agent_executor, name="CEO")
cto_node = functools.partial(worker_node, agent=cto_agent_executor, name="CTO") 
cfo_node = functools.partial(worker_node, agent=cfo_agent_executor, name="CFO")
coo_node = functools.partial(worker_node, agent=coo_agent_executor, name="COO")

# --- Enhanced Supervisor Node ---
def supervisor_node(state):
    """Enhanced supervisor with search-aware routing"""
    messages = state.get("messages", [])
    message_count = state.get("message_count", 0)
    last_speaker = state.get("last_speaker", "")
    agent_participation = state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False})
    pending_questions = state.get("pending_questions", [])
    topics_discussed = state.get("topics_discussed", [])
    
    # Check if we already have a final report
    if messages:
        last_message = messages[-1].content
        if check_for_final_report(last_message):
            return {"next": "FINISH"}
    
    # RULE 1: Force final report at message 10+ when all participated
    if message_count >= 10 and all(agent_participation.values()):
        return {"next": "CEO"}
    
    # RULE 2: Ensure each agent participates at least once
    for agent in ["CEO", "CTO", "CFO", "COO"]:
        if not agent_participation[agent] and agent != last_speaker:
            return {"next": agent}
    
    # RULE 3: Handle pending questions (but not to last speaker)
    if pending_questions and message_count < 10:
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
    
    # RULE 4: Topic-based intelligent routing
    if topics_discussed and message_count < 10:
        latest_topics = topics_discussed[-3:] if len(topics_discussed) >= 3 else topics_discussed
        
        # Route based on topic expertise
        if "technical" in latest_topics and last_speaker != "CTO":
            return {"next": "CTO"}
        elif "financial" in latest_topics and last_speaker != "CFO":
            return {"next": "CFO"}
        elif "operations" in latest_topics and last_speaker != "COO":
            return {"next": "COO"}
        elif "strategy" in latest_topics and last_speaker != "CEO":
            return {"next": "CEO"}
    
    # RULE 5: Dynamic round-robin routing (excluding last speaker)
    if message_count < 10:
        available_agents = [agent for agent in ["CEO", "CTO", "CFO", "COO"] if agent != last_speaker]
        if available_agents:
            next_index = message_count % len(available_agents)
            return {"next": available_agents[next_index]}
    
    # RULE 6: Safety fallback
    return {"next": "FINISH"}

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
        "agent_participation": {"CEO": False, "CTO": False, "CFO": False, "COO": False}
    }
    
    print("\n--- Starting Enhanced AI Startup Consultation ---")
    print("🔍 Agents now have access to real-time market data and industry insights\n")
    
    # Add recursion limit to prevent infinite loops
    config = {"recursion_limit": 50}
    
    for output in app.stream(initial_state, config=config):
        for key, value in output.items():
            if key != "__end__" and "messages" in value and value['messages']:
                agent_name = key.upper()
                agent_message = value['messages'][-1].content
                print(f"--- {agent_name} ---")
                print(agent_message)
                print()
    
    print("--- Enhanced Consultation Finished ---")
    print("💡 Your consultation included real-time market research and data-driven insights!")

if __name__ == "__main__":
    main()
