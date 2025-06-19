import streamlit as st
from main_v3 import app # Import your compiled LangGraph app [1]
from langchain_core.messages import HumanMessage

# Define emoji icons for each role
AVATAR_ICONS = {
    "user": "👤",
    "assistant": "🤖", # General assistant/welcome message
    "ceo": "💼",       # Crown or person in suit for CEO [7][10][11]
    "cto": "💻",       # Laptop or gear for CTO [5]
    "cfo": "💰",       # Money bag or chart for CFO [3][8]
    "coo": "⚙️",       # Gear or handshake for COO [4]
    "final_report": "✅" # Checkmark for the final report
}

# Set the page title and layout to wide mode
st.set_page_config(page_title="AI Startup Consultancy", layout="wide")
st.title("🤖 AI Startup Consultancy Firm")

# Add a sidebar for context and instructions
with st.sidebar:
    st.header("About This App")
    st.write("This application simulates a multi-agent startup consultancy meeting.")
    st.write("Enter your business idea, and our AI board (CEO, CTO, CFO, COO) will discuss, analyze, and provide a comprehensive report.")
    st.markdown("---")
    st.subheader("How It Works")
    st.write("1. **Submit Idea:** Enter your business concept in the input box.")
    st.write("2. **Board Discussion:** Observe as the AI agents (CEO, CTO, CFO, COO) engage in a structured conversation, researching and refining the idea.")
    st.write("3. **Final Report:** The CEO will synthesize the discussion into a comprehensive final report.")
    st.markdown("---")
    st.caption("Powered by LangGraph and Streamlit")


# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial welcome message from the assistant
    st.session_state.messages.append({"role": "assistant", "content": "Welcome! Please enter your business idea to kick off our board meeting."})

# Display prior chat messages from session state
for message in st.session_state.messages:
    # Determine the role for display and its corresponding avatar
    current_role = message["role"]
    display_name = current_role.upper() # Default to uppercase for agents

    if current_role == "user":
        display_name = "You"
        avatar_icon = AVATAR_ICONS["user"]
    elif current_role == "assistant":
        display_name = "Assistant"
        avatar_icon = AVATAR_ICONS["assistant"]
    elif current_role == "final_report": # Handle the final_report specifically
        display_name = "CEO (FINAL REPORT)"
        avatar_icon = AVATAR_ICONS["final_report"]
    else: # This applies to 'ceo', 'cto', 'cfo', 'coo'
        avatar_icon = AVATAR_ICONS.get(current_role, "💬") # Use a default if role not found
        # LangGraph node names are lowercase (e.g., 'ceo'), so match them to AVATAR_ICONS keys.

    with st.chat_message(name=display_name, avatar=avatar_icon):
        st.markdown(message["content"])

# Handle user input for the business idea
if prompt := st.chat_input("What business idea would you like our board to consult on?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message(name="You", avatar=AVATAR_ICONS["user"]):
        st.markdown(prompt)

    # Prepare the initial message for the agentic workflow.
    initial_messages = [HumanMessage(content=f"Analyze the following business idea and delegate tasks to the team. Business Idea: {prompt}")]

    # Use a spinner to indicate that the agents are working
    with st.spinner("The AI board is in session, discussing your idea..."):
        # Stream the output from the LangGraph application
        stream = app.stream({"messages": initial_messages}) # app from main_v3.py [1]

        # Process and display each message from the stream
        for output in stream:
            for key, value in output.items():
                if key != "__end__" and "messages" in value:
                    # 'key' directly comes from the node name in LangGraph (e.g., "CEO", "CTO" from agent.py `name` param) [1]
                    # Convert to lowercase for matching with AVATAR_ICONS keys.
                    agent_role_key = key.lower()
                    
                    # Determine the display name and avatar for the agent
                    display_name = key.upper() # Display 'CEO', 'CTO', etc.
                    if agent_role_key == "final_report":
                        display_name = "CEO (FINAL REPORT)" # Specific label for the final output
                        avatar_icon = AVATAR_ICONS["final_report"]
                    else:
                        avatar_icon = AVATAR_ICONS.get(agent_role_key, "💬") # Get icon, default if not found

                    agent_message_content = value['messages'][-1].content

                    # Display the agent's message in a chat bubble
                    with st.chat_message(name=display_name, avatar=avatar_icon):
                        st.markdown(agent_message_content)

                    # Add the agent's message to the session state for persistence
                    st.session_state.messages.append({"role": agent_role_key, "content": agent_message_content})