# streamlit_app.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Import the function that creates our LangGraph app
from main import get_graph_app

# --- App Configuration and Title ---
st.set_page_config(page_title="Multi-Agent Startup Platform", layout="wide")
st.title("Multi-Agent Startup Platform")
st.write("Welcome! Pitch your business idea and get a comprehensive business plan from our C-suite agents.")

# --- Graph Initialization ---
# Use st.cache_resource to ensure the graph is loaded only once
@st.cache_resource
def load_graph():
    return get_graph_app()

app = load_graph()

# --- Session State Management ---
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="What is your business idea?")]

# --- Chat Interface ---
# Display existing messages from session state
for msg in st.session_state.messages:
    # Determine the role for the chat message
    if isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, HumanMessage):
        role = msg.name if msg.name and msg.name != "user" else "user"
    
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle user input
if prompt := st.chat_input("Your business idea..."):
    # Create a HumanMessage for the user's input
    user_message = HumanMessage(content=prompt, name="user")
    
    # Add user message to session state and display it
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Agent Workflow Execution ---
    # Display a spinner while the agents are working
    with st.spinner("The agents are discussing your idea..."):
        # The input for the graph is the current list of all messages
        inputs = {"messages": st.session_state.messages}
        
        # Stream the output from the LangGraph application
        for output in app.stream(inputs):
            for key, value in output.items():
                if key != "__end__":
                    if 'messages' in value:
                        # An agent has responded. Get the message.
                        agent_message = value['messages'][-1]
                        
                        # Add the agent's message to the session state
                        st.session_state.messages.append(agent_message)
                        
                        # Display the new message from the agent in the chat
                        with st.chat_message(agent_message.name):
                            st.markdown(agent_message.content)


