from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_supervisor_chain(llm, members):
    """
    Enhanced supervisor with context-aware routing for dynamic discussions.
    """
    system_prompt = """You are moderating a startup boardroom discussion with: {members} (Sarah=CEO, Mike=CTO, Jennifer=CFO, Tom=COO).

    **Meeting Phases & Routing Logic:**

    **Phase 1 - Initial Round (0-4 messages):**
    - Route systematically: CEO → CTO → CFO → COO to get initial perspectives
    - Each agent should provide their domain-specific analysis

    **Phase 2 - Dynamic Discussion (5-15 messages):**
    - Route based on conversation context and direct questions
    - Look for: "Mike, what do you think...", "Jennifer, how does this affect...", "Tom, can we execute..."
    - Route to agent whose expertise is most relevant to the last point made
    - Encourage cross-pollination of ideas

    **Phase 3 - Synthesis (15+ messages):**
    - When discussion feels complete AND no pending questions, route to CEO for "FINISH"

    **Dynamic Routing Priorities:**
    1. **Direct Questions:** If someone asks "Mike, ...", route to CTO
    2. **Technical Topics:** Architecture, development, scalability → CTO
    3. **Financial Topics:** Costs, funding, projections → CFO  
    4. **Operational Topics:** Hiring, timeline, execution → COO
    5. **Strategic Topics:** Vision, market, synthesis → CEO

    **Route to FINISH only when:**
    - Substantial discussion has occurred (15+ messages)
    - No direct questions are pending
    - All major aspects have been covered
    - Natural conversation conclusion is reached

    Current conversation context: Analyze the last 2-3 messages to determine who should respond next."""

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next agent to continue the discussion or finish.",
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
            "Based on the conversation above, who should speak next? Consider:\n"
            "- Are there direct questions to specific people?\n"
            "- What expertise is most relevant to continue this discussion?\n"
            "- Is this a natural place for the CEO to synthesize? (Only if 15+ messages and discussion feels complete)\n"
            "Select one of: {options}"
        ),
    ]).partial(options=str(options), members=", ".join(members))

    return prompt | llm.bind_tools(tools=[function_def], tool_choice="route")
