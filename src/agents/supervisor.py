from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_supervisor_chain(llm, members):
    """Enhanced supervisor with better conversation flow and stagnation detection"""

    system_prompt = """You are moderating a startup consultation meeting with: {members} (Sarah=CEO, Mike=CTO, Jennifer=CFO, Tom=COO).

**CRITICAL CONVERSATION RULES:**

1. **NEVER route back to the agent who just spoke** (unless it's final report time)

2. **Detect conversation stagnation**: If the last 3 messages are too similar or repetitive, force a perspective shift by routing to the least active agent

3. **Route to FINISH only when**: Comprehensive analysis is complete (15+ messages) AND no pending questions exist AND a final report has been provided

4. **Prioritize natural conversation flow**: Route based on who would naturally respond next in a real meeting

**ROUTING PRIORITIES:**

1. **Direct Questions**: If someone asks "Mike, what do you think..." → route to CTO (if CTO didn't just speak)

2. **Topic Expertise**: Route based on conversation context and who hasn't spoken recently
   - Technical topics → CTO (if not last speaker)
   - Financial topics → CFO (if not last speaker)  
   - Operations topics → COO (if not last speaker)
   - Strategy/synthesis → CEO (if not last speaker)

3. **Conversation Balance**: Ensure each agent participates and no one dominates

4. **Stagnation Prevention**: If conversation becomes repetitive, route to least active agent to inject new perspective

**PHASE MANAGEMENT:**
- Initial (0-4 messages): Each agent provides domain analysis
- Discussion (5-15 messages): Dynamic interaction based on questions, reactions, and expertise
- Synthesis (15+ messages): CEO provides final comprehensive report

**QUALITY INDICATORS:**
- Look for agents building on each other's points
- Ensure agents are reacting to, not just adding to, the discussion
- Route to create natural conversation flow, not just information sharing

Current context: Analyze the last 2-3 messages to determine optimal next speaker who will advance the conversation naturally."""

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
        },
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Based on the conversation, who should speak next? Consider:\n"
            "- Who just spoke? (DON'T route back to them unless final report)\n"
            "- Are there specific questions for team members?\n" 
            "- What expertise is needed to continue the discussion?\n"
            "- Are the last few messages too similar? (Force perspective shift)\n"
            "- Who would naturally respond next in a real meeting?\n"
            "- Is this ready for final synthesis? (Only if 15+ messages AND discussion feels complete)\n"
            "Choose from: {options}"
        ),
    ]).partial(options=str(options), members=", ".join(members))

    return prompt | llm.bind_tools(tools=[function_def], tool_choice="route")
