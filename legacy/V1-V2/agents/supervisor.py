# agents/supervisor.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_supervisor_chain(llm, members):
    """
    Supervisor agent is responsible for routing tasks to the correct worker agent.
    """
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: {members}. Given the user request, determine which"
        " worker should act next. Each worker will perform a"
        " task and respond with their results and status. When the final report is ready,"
        " respond with 'FINISH'."
    )

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next worker to act or finish.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))
    return (
        prompt
        | llm.bind_tools(tools=[function_def], tool_choice="route")
    )
