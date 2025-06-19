# main.py

import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
import functools
import json

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

# Load variables from .env file
load_dotenv()

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next: str

def get_graph_app():
    """
    Creates and compiles the multi-agent LangGraph application.
    This function is now importable by other scripts, like our Streamlit app.
    """
    # --- Agent and Tool Creation ---
    tools = [TavilySearch(max_results=3)]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Create the worker agents and the supervisor
    ceo_agent_executor = create_ceo_agent(llm, tools)
    cfo_agent_executor = create_cfo_agent(llm, tools)
    cto_agent_executor = create_cto_agent(llm, tools)
    coo_agent_executor = create_coo_agent(llm, tools)

    members = ["CEO", "CTO", "CFO", "COO"]
    supervisor_chain = create_supervisor_chain(llm, members)

    # --- Node and Graph Definition ---
    def worker_node(state, agent, name):
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}

    ceo_node = functools.partial(worker_node, agent=ceo_agent_executor, name="CEO")
    cto_node = functools.partial(worker_node, agent=cto_agent_executor, name="CTO")
    cfo_node = functools.partial(worker_node, agent=cfo_agent_executor, name="CFO")
    coo_node = functools.partial(worker_node, agent=coo_agent_executor, name="COO")

    def supervisor_node(state):
        result = supervisor_chain.invoke(state)
        if result.tool_calls:
            call = result.tool_calls[0]
            if call['name'] == "route":
                arguments = call['args']
                next_agent = arguments["next"]
                return {"next": next_agent}
        return {"next": "FINISH"}

    workflow = StateGraph(AgentState)
    workflow.add_node("CEO", ceo_node)
    workflow.add_node("CTO", cto_node)
    workflow.add_node("CFO", cfo_node)
    workflow.add_node("COO", coo_node)
    workflow.add_node("supervisor", supervisor_node)

    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    workflow.set_entry_point("supervisor")
    
    return workflow.compile()

# This part allows you to still run main.py directly for testing if needed
if __name__ == "__main__":
    app = get_graph_app()
    idea = input("Please enter your business idea: ")
    initial_messages = [HumanMessage(content=f"Here is the business idea: {idea}")]
    
    print("\n--- Starting Startup Consultation ---\n")
    
    for output in app.stream({"messages": initial_messages}):
        for key, value in output.items():
            if key != "__end__":
                print(f"--- Node: {key} ---")
                if 'messages' in value:
                    print(value['messages'][-1].content)
                else:
                    print(f"Routing to: {value.get('next')}")
                print("\n")

    print("--- Consultation Finished ---")

