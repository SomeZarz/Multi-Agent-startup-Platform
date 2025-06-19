# main_v3.py (Corrected)

import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
import functools
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

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next: str

# --- Agent and Tool Creation ---
tools = [TavilySearch(max_results=3)]
# Ensure you are using a capable model like gpt-4o for best results
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create worker agents
ceo_agent_executor = create_ceo_agent(llm, tools)
cfo_agent_executor = create_cfo_agent(llm, tools)
cto_agent_executor = create_cto_agent(llm, tools)
coo_agent_executor = create_coo_agent(llm, tools)

# Create the supervisor agent
members = ["CEO", "CTO", "CFO", "COO"]
supervisor_chain = create_supervisor_chain(llm, members)

# --- Node and Graph Definition ---

# Helper function to create a worker node
def worker_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Define nodes for each worker agent
ceo_node = functools.partial(worker_node, agent=ceo_agent_executor, name="CEO")
cto_node = functools.partial(worker_node, agent=cto_agent_executor, name="CTO")
cfo_node = functools.partial(worker_node, agent=cfo_agent_executor, name="CFO")
coo_node = functools.partial(worker_node, agent=coo_agent_executor, name="COO")

# The supervisor node routes tasks
def supervisor_node(state):
    result = supervisor_chain.invoke(state)
    # Use the robust 'tool_calls' attribute for modern langchain versions
    if hasattr(result, 'tool_calls') and result.tool_calls:
        call = result.tool_calls[0]
        if call["name"] == "route":
            next_agent = call['args']['next']
            return {"next": next_agent}
    return {"next": "FINISH"} # Default to FINISH if no route is found

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("CEO", ceo_node)
workflow.add_node("CTO", cto_node)
workflow.add_node("CFO", cfo_node)
workflow.add_node("COO", coo_node)
workflow.add_node("supervisor", supervisor_node)

# --- **FIX**: Add a new node specifically for the final report ---
# This node also uses the CEO agent, as the CEO is responsible for the final synthesis.
workflow.add_node("final_report", ceo_node)

# Add edges from each worker back to the supervisor
for member in members:
    workflow.add_edge(member, "supervisor")

# The conditional edge maps the supervisor's choice to the next node
conditional_map = {k: k for k in members}
# --- **FIX**: Route "FINISH" to the new final_report node instead of END ---
conditional_map["FINISH"] = "final_report"

workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# --- **FIX**: The final_report node is now the last step before ending ---
workflow.add_edge("final_report", END)

workflow.set_entry_point("supervisor")

# Compile the graph
app = workflow.compile()

# --- Graph Execution ---
def main():
    idea = input("Please enter your business idea: ")
    initial_messages = [HumanMessage(content=f"Your task is to act as a startup consulting firm. Start by having the CEO provide an initial analysis of the following business idea and delegate tasks to the team. Business Idea: {idea}")]
    print("\n--- Starting Startup Consultation ---\n")
    
    # Stream the output from the graph
    for output in app.stream({"messages": initial_messages}):
        for key, value in output.items():
            if key != "__end__":
                # Check for the 'messages' key to avoid errors from supervisor outputs
                if "messages" in value:
                    agent_name = key.upper()
                    # The final report node should also be clearly labeled
                    if agent_name == "FINAL_REPORT":
                        agent_name = "CEO (FINAL REPORT)"
                    
                    agent_message = value['messages'][-1].content
                    print(f"--- {agent_name} ---")
                    print("\n")

    print("--- Consultation Finished ---")

if __name__ == "__main__":
    main()


