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
    agent_participation: dict
    conversation_context: str  # New: recent conversation context

# --- Agent and Tool Creation ---
tools = [TavilySearch(max_results=3)]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # Increased temperature for more natural responses

# Create worker agents
ceo_agent_executor = create_ceo_agent(llm, tools)
cfo_agent_executor = create_cfo_agent(llm, tools)
cto_agent_executor = create_cto_agent(llm, tools)
coo_agent_executor = create_coo_agent(llm, tools)

# Create the supervisor agent
members = ["CEO", "CTO", "CFO", "COO"]
supervisor_chain = create_supervisor_chain(llm, members)

# --- Enhanced Helper Functions ---
def extract_conversational_context(messages, current_agent):
    """Extract recent conversational context for more natural responses"""
    if len(messages) < 2:
        return ""
    
    # Get last 2-3 messages for context
    recent_messages = messages[-3:]
    context_parts = []
    
    for msg in recent_messages:
        speaker = getattr(msg, 'name', 'Unknown')
        content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
        context_parts.append(f"**{speaker}**: {content_preview}")
    
    return "\n".join(context_parts)

def detect_direct_questions(content, current_agent):
    """Detect questions directed at specific agents"""
    questions = []
    
    # Agent name patterns
    agent_patterns = {
        "CEO": ["Sarah", "CEO"],
        "CTO": ["Mike", "CTO"], 
        "CFO": ["Jennifer", "CFO"],
        "COO": ["Tom", "COO"]
    }
    
    for agent, names in agent_patterns.items():
        if agent != current_agent:
            for name in names:
                # Look for direct questions
                patterns = [
                    rf"{name},\s*([^?]*\?)",
                    rf"{name}\s*[,-]\s*([^?]*\?)",
                    rf"@{name}\s*([^?]*\?)"
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        questions.append(f"{agent}: {match.strip()}")
    
    return questions

def extract_topics_discussed(content):
    """Enhanced topic extraction"""
    topics = []
    content_lower = content.lower()
    
    topic_keywords = {
        "technical": ["technology", "tech stack", "architecture", "development", "MVP", "API", "database", "scalability"],
        "financial": ["budget", "funding", "revenue", "costs", "financial", "money", "investment", "profit", "burn rate"],
        "market": ["market", "competition", "customers", "users", "TAM", "SAM", "segments", "competitive advantage"],
        "operations": ["operations", "hiring", "timeline", "execution", "team", "go-to-market", "launch", "processes"],
        "strategy": ["strategy", "vision", "goals", "planning", "roadmap", "positioning", "differentiation"]
    }
    
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
        "FINAL RECOMMENDATION:",
        "COMPREHENSIVE ANALYSIS",
        "FINAL SYNTHESIS"
    ]
    
    has_indicator = any(indicator in content_upper for indicator in final_report_indicators)
    is_substantial = len(content) > 500
    
    return has_indicator and is_substantial

# --- Enhanced Worker Node with Conversational Context ---
def worker_node(state, agent, name):
    """Enhanced worker node with conversational context"""
    
    last_speaker = state.get("last_speaker", "")
    message_count = state.get("message_count", 0)
    agent_participation = state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False})
    messages = state.get("messages", [])
    
    # Check for final report requirement
    is_final_report_time = (
        message_count >= 12 and
        all(agent_participation.values()) and
        name == "CEO"
    )
    
    # Apply "no consecutive speaker" rule EXCEPT for final report
    if last_speaker == name and not is_final_report_time:
        return state
    
    # Generate conversational context
    conversation_context = extract_conversational_context(messages, name)
    
    # Prepare state for agent
    if is_final_report_time:
        # Force final report
        modified_state = dict(state)
        modified_state["messages"] = state["messages"] + [
            HumanMessage(
                content="SYSTEM: Please provide the FINAL REPORT summarizing our comprehensive analysis.",
                name="system"
            )
        ]
        result = agent.invoke(modified_state)
    else:
        # Add conversational context for natural responses
        if conversation_context and message_count > 1:
            context_message = HumanMessage(
                content=f"RECENT CONVERSATION CONTEXT:\n{conversation_context}\n\nNow respond naturally to continue this conversation:",
                name="context_system"
            )
            
            modified_state = dict(state)
            modified_state["messages"] = state["messages"] + [context_message]
            result = agent.invoke(modified_state)
        else:
            result = agent.invoke(state)
    
    content = result["output"]
    
    # Extract insights from response
    questions = detect_direct_questions(content, name)
    new_topics = extract_topics_discussed(content)
    existing_topics = state.get("topics_discussed", [])
    combined_topics = list(set(existing_topics + new_topics))
    
    # Update agent participation
    updated_participation = dict(state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False}))
    updated_participation[name] = True
    
    # Increment message count and determine phase
    new_count = state.get("message_count", 0) + 1
    
    if new_count <= 4:
        phase = "initial"
    elif new_count <= 12:
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
        "conversation_context": conversation_context
    }

# Define nodes for each worker agent
ceo_node = functools.partial(worker_node, agent=ceo_agent_executor, name="CEO")
cto_node = functools.partial(worker_node, agent=cto_agent_executor, name="CTO")
cfo_node = functools.partial(worker_node, agent=cfo_agent_executor, name="CFO")
coo_node = functools.partial(worker_node, agent=coo_agent_executor, name="COO")

# --- Enhanced Supervisor Node ---
def supervisor_node(state):
    """Enhanced supervisor with natural conversation flow"""
    
    messages = state.get("messages", [])
    message_count = state.get("message_count", 0)
    last_speaker = state.get("last_speaker", "")
    agent_participation = state.get("agent_participation", {"CEO": False, "CTO": False, "CFO": False, "COO": False})
    pending_questions = state.get("pending_questions", [])
    
    # Check for final report completion
    if messages:
        last_message = messages[-1].content
        if check_for_final_report(last_message):
            return {"next": "FINISH"}
    
    # Force final report at appropriate time
    if message_count >= 12 and all(agent_participation.values()):
        return {"next": "CEO"}
    
    # Ensure all agents participate initially  
    for agent in ["CEO", "CTO", "CFO", "COO"]:
        if not agent_participation[agent] and agent != last_speaker:
            return {"next": agent}
    
    # Handle direct questions/references
    if pending_questions and message_count < 12:
        latest_question = pending_questions[-1]
        for agent in ["CEO", "CTO", "CFO", "COO"]:
            if f"{agent}:" in latest_question and agent != last_speaker:
                return {"next": agent}
    
    # Natural conversation flow based on content
    if messages and message_count < 12:
        last_content = messages[-1].content.lower()
        available_agents = [agent for agent in ["CEO", "CTO", "CFO", "COO"] if agent != last_speaker]
        
        # Topic-based routing
        if any(word in last_content for word in ["technical", "technology", "development", "architecture"]):
            if "CTO" in available_agents:
                return {"next": "CTO"}
        elif any(word in last_content for word in ["financial", "budget", "revenue", "funding", "cost"]):
            if "CFO" in available_agents:
                return {"next": "CFO"}  
        elif any(word in last_content for word in ["operations", "execution", "timeline", "hiring", "team"]):
            if "COO" in available_agents:
                return {"next": "COO"}
        elif any(word in last_content for word in ["strategy", "vision", "market", "competitive"]):
            if "CEO" in available_agents:
                return {"next": "CEO"}
        
        # Default round-robin for available agents
        if available_agents:
            next_index = message_count % len(available_agents)
            return {"next": available_agents[next_index]}
    
    # Fallback
    return {"next": "FINISH"}

# --- Graph Definition ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("CEO", ceo_node)
workflow.add_node("CTO", cto_node)
workflow.add_node("CFO", cfo_node)
workflow.add_node("COO", coo_node)
workflow.add_node("supervisor", supervisor_node)

# Add edges from each worker back to supervisor
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
    
    initial_messages = [HumanMessage(
        content=f"Analyze the following business idea and provide comprehensive consultation. Business Idea: {idea}"
    )]
    
    initial_state = {
        "messages": initial_messages,
        "discussion_phase": "initial",
        "topics_discussed": [],
        "pending_questions": [],
        "message_count": 0,
        "last_speaker": "",
        "agent_participation": {"CEO": False, "CTO": False, "CFO": False, "COO": False},
        "conversation_context": ""
    }
    
    print("\n--- Starting Natural Startup Consultation ---\n")
    
    # Increased recursion limit for more natural conversation
    config = {"recursion_limit": 60}
    
    for output in app.stream(initial_state, config=config):
        for key, value in output.items():
            if key != "__end__" and "messages" in value and value['messages']:
                agent_name = key.upper()
                agent_message = value['messages'][-1].content
                print(f"--- {agent_name} ---")
                print(agent_message)
                print()
    
    print("--- Consultation Finished ---")

if __name__ == "__main__":
    main()

