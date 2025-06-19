from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_supervisor_chain(llm, members):
    """Enhanced supervisor with natural conversation flow"""
    
    system_prompt = """You are moderating a dynamic startup consultation with: {members} (Sarah=CEO, Mike=CTO, Jennifer=CFO, Tom=COO).

**Natural Conversation Rules:**

1. **NO BACK-TO-BACK**: Never route to the agent who just spoke
2. **FOLLOW THE ENERGY**: Route based on natural conversation flow, not rigid patterns
3. **FINISH WISELY**: Only route to FINISH when you have 15+ meaningful exchanges AND natural conclusion

**Priority Routing Logic:**

1. **Direct Questions/Names**: If someone asks "Mike, what do you think..." → route to CTO (unless CTO just spoke)
2. **Topic Expertise**: 
   - Technical discussions → CTO
   - Financial implications → CFO  
   - Execution/operations → COO
   - Strategy/vision/synthesis → CEO
3. **Natural Flow**: Let agents build on each other's ideas
4. **Question Chains**: If someone asks a follow-up, route to the person being asked

**Conversation Phases:**
- **Startup (1-4 messages)**: Get each agent's initial perspective
- **Dynamic Discussion (5-14 messages)**: Follow natural conversation flow
- **Synthesis (15+ messages)**: Move toward CEO for final comprehensive report

**Advanced Flow Detection:**
- Look for conversation threads and continue them
- Detect when agents reference each other and facilitate follow-up
- Allow natural interruptions when someone is directly addressed
- Encourage cross-functional collaboration

Current context: Analyze the last 2-3 messages to determine who should naturally speak next based on conversation flow, not just turn-taking."""

    options = ["FINISH"] + members
    
    function_def = {
        "name": "route",
        "description": "Select the next agent based on natural conversation flow",
        "parameters": {
            "title": "routeSchema", 
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                }
            },
            "required": ["next"],
        }
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system", 
            "Based on the conversation flow, who should naturally speak next?\n"
            "Consider:\n"
            "- Who just spoke? (NEVER route back to them)\n" 
            "- Are there direct questions or references to specific team members?\n"
            "- What topic expertise is needed to continue this thread?\n"
            "- Is someone building on another person's idea?\n"
            "- Does the conversation feel ready for final synthesis? (15+ messages AND natural conclusion)\n"
            "Choose from: {options}"
        ),
    ]).partial(options=str(options), members=", ".join(members))
    
    return prompt | llm.bind_tools(tools=[function_def], tool_choice="route")


