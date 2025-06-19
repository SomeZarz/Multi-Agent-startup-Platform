from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_supervisor_chain(llm, members):
    """Enhanced supervisor with better conversation flow control"""
    
    system_prompt = """You are moderating a startup consultation meeting with: {members} (Sarah=CEO, Mike=CTO, Jennifer=CFO, Tom=COO).

**Critical Rules:**
1. NEVER route back to the agent who just spoke
2. Route to FINISH only when comprehensive analysis is complete (15+ messages) AND no pending questions exist
3. Prioritize agents who haven't spoken recently

**Routing Priorities:**
1. **Direct Questions**: If someone asks "Mike, what do you think..." → route to CTO (if CTO didn't just speak)
2. **Topic Expertise**: Route based on conversation context
   - Technical topics → CTO
   - Financial topics → CFO  
   - Operations topics → COO
   - Strategy/synthesis → CEO
3. **Natural Flow**: Encourage cross-team discussion

**Phase Management:**
- Initial (0-4 messages): Each agent provides domain analysis
- Discussion (5-15 messages): Dynamic interaction based on questions and expertise
- Synthesis (15+ messages): CEO provides final comprehensive report

Current context: Look at the last 2-3 messages to determine optimal next speaker."""

    options = ["FINISH"] + members
    
    function_def = {
        "name": "route",
        "description": "Select the next agent or finish the discussion",
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
            "Based on the conversation, who should speak next? Consider:\n"
            "- Who just spoke? (DON'T route back to them)\n"
            "- Are there specific questions for team members?\n"
            "- What expertise is needed to continue the discussion?\n"
            "- Is this ready for final synthesis? (Only if 15+ messages AND discussion feels complete)\n"
            "Choose from: {options}"
        ),
    ]).partial(options=str(options), members=", ".join(members))

    return prompt | llm.bind_tools(tools=[function_def], tool_choice="route")

