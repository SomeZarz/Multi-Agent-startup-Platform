import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
# Updated import for the new Tavily Search tool
from langchain_tavily import TavilySearch

# Import agents
from agents.ceo import create_ceo_agent
from agents.cfo import create_cfo_agent
from agents.cto import create_cto_agent
from agents.coo import create_coo_agent

# Load variables from .env file
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# Updated tool instantiation with the new TavilySearch class
tools = [TavilySearch(max_results=3)]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Agent Creation
ceo_agent_executor = create_ceo_agent(llm, tools)
cfo_agent_executor = create_cfo_agent(llm, tools)
cto_agent_executor = create_cto_agent(llm, tools)
coo_agent_executor = create_coo_agent(llm, tools)

# --- Node Functions ---
def ceo_node(state):
    response = ceo_agent_executor.invoke(state)
    return {"messages": [HumanMessage(content=response["output"], name="CEO")]}

def cto_node(state):
    response = cto_agent_executor.invoke(state)
    return {"messages": [HumanMessage(content=response["output"], name="CTO")]}

def cfo_node(state):
    response = cfo_agent_executor.invoke(state)
    return {"messages": [HumanMessage(content=response["output"], name="CFO")]}

def coo_node(state):
    response = coo_agent_executor.invoke(state)
    return {"messages": [HumanMessage(content=response["output"], name="COO")]}

# --- Router Function ---
def router(state):
    last_message = state["messages"][-1]
    
    # If the CEO has generated the final report, end the conversation.
    if "FINAL REPORT:" in last_message.content:
        return END
    
    # Route to the appropriate agent based on who spoke last.
    if last_message.name == "CEO":
        return "CTO"
    elif last_message.name == "CTO":
        return "CFO"
    elif last_message.name == "CFO":
        return "COO"
    elif last_message.name == "COO":
        return "CEO" # After COO, CEO critiques.
    else:
        # Fallback route for the initial message
        return "CEO"

# --- Graph Definition and Execution ---
def main():
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("CEO", ceo_node)
    workflow.add_node("CTO", cto_node)
    workflow.add_node("CFO", cfo_node)
    workflow.add_node("COO", coo_node)

    workflow.set_entry_point("CEO")

    # CORRECTED: Each conditional edge call is now a separate, valid statement.
    # Each mapping now includes END: END to handle graceful termination.
    workflow.add_conditional_edges(
        "CEO",
        router,
        {"CTO": "CTO", "CEO": "CEO", END: END}
    )
    workflow.add_conditional_edges(
        "CTO",
        router,
        {"CFO": "CFO", END: END}
    )
    workflow.add_conditional_edges(
        "CFO",
        router,
        {"COO": "COO", END: END}
    )
    workflow.add_conditional_edges(
        "COO",
        router,
        {"CEO": "CEO", END: END}
    )

    app = workflow.compile()
    
    idea = input("Please enter your business idea: ")
    initial_messages = [HumanMessage(content=f"Here is the business idea: {idea}")]
    
    print("\n--- Starting Startup Consultation ---\n")
    
    for output in app.stream({"messages": initial_messages}):
        for key, value in output.items():
            if key != "__end__":
                agent_name = key
                agent_message = value['messages'][-1].content
                print(f"--- {agent_name} ---")
                print(agent_message)
                print("\n")
    
    # CORRECTED: Moved the "Finished" message outside the loop.
    print("--- Consultation Finished ---")

if __name__ == "__main__":
    main()
