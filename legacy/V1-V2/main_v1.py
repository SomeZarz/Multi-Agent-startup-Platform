import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Import agent 
from agents.ceo import create_ceo_agent
from agents.cfo import create_cfo_agent
from agents.cto import create_cto_agent
from agents.coo import create_coo_agent

# Loadvariables from .env file
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] #shared state for all agents


tools = [TavilySearchResults(max_results=3)] #tavily search tool


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) #gpt-3.5-turbo model with low temperature for all models

# Agent Creation
ceo_agent_executor = create_ceo_agent(llm, tools)
cfo_agent_executor = create_cfo_agent(llm, tools)
cto_agent_executor = create_cto_agent(llm, tools)
coo_agent_executor = create_coo_agent(llm, tools)

# Node Functions
# turn by turn response node for each agent
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


def router(state):
    last_message = state["messages"][-1] #determines the best agent to act based on conversation history
    
    # If the CEO has generated the final report, end the conversation.
    if "FINAL REPORT:" in last_message.content:
        return END
    
    # Route to the appropriate agent based on who had the last convo
    if last_message.name == "CEO":
        return "CTO"
    elif last_message.name == "CTO":
        return "CFO"
    elif last_message.name == "CFO":
        return "COO"
    elif last_message.name == "COO":
        
        return "CEO" #ceo critisises after agents are done.
    else:
        
        return END #or not?

# --- Graph Definition and Execution ---
def main():
    # Define the graph structure
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("CEO", ceo_node)
    workflow.add_node("CTO", cto_node)
    workflow.add_node("CFO", cfo_node)
    workflow.add_node("COO", coo_node)

    
    workflow.set_entry_point("CEO") # the graph entry point

    # Add conditional edges to route the conversation
    workflow.add_conditional_edges(
        "CEO",
        router,
        {"CTO": "CTO", "CEO": "CEO"} # The CEO can talk to the CTO or continue talking to itself (for critique phase)
    )
    workflow.add_conditional_edges(
        "CTO",
        router,
        {"CFO": "CFO"}
    )
    workflow.add_conditional_edges(
        "CFO",
        router,
        {"COO": "COO"}
    )
    workflow.add_conditional_edges(
        "COO",
        router,
        {"CEO": "CEO"}
    )

    
    app = workflow.compile() #executable app on terminal

    
    idea = input("Please enter your business idea: ") #user inputs idea
    
    
    initial_messages = [HumanMessage(content=f"Here is the business idea: {idea}")] #prepare the message for langgraph
    
    print("\n--- Starting Startup Consultation ---\n")
    
    
    for output in app.stream({"messages": initial_messages}): #show live convo using stream
        
        # The key of the output dictionary is the name of the node that just ran
        for key, value in output.items():
            if key != "__end__":
                agent_name = key
                agent_message = value['messages'][-1].content
                print(f"--- {agent_name} ---")
                print(agent_message)
                print("\n")

    print("--- Consultation Finished ---")

if __name__ == "__main__":
    main()