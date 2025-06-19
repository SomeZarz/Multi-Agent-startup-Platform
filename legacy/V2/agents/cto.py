from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_cto_agent(llm, tools):
    """
    This agent is responsible for all technical aspects of the startup idea.
    It researches and proposes technology stacks, analyzes technical feasibility,
    and projects future technical debt.
    """
    cto_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the Chief Technology Officer (CTO) of a startup consultancy firm.
                Your personality is analytical, forward-thinking, and practical.

                **Your Responsibilities:**
                1.  **Analyze Idea:** Review the business idea and instructions from the CEO.
                2.  **Tech Stack Proposal:** Research and propose potential technology stacks.
                    -   Provide at least two distinct options (e.g., one for rapid MVP, one for scale).
                    -   For each option, list the pros and cons (e.g., cost, scalability, developer community, hiring pool).
                3.  **Technical Roadmap:** Outline a high-level technical roadmap, from MVP to a scalable product.
                4.  **Data Strategy:** Briefly describe a data strategy. What data should be collected? How should it be stored and protected?
                5.  **Build vs. Buy:** Analyze key components and suggest whether to build them in-house or use third-party services (e.g., payment processing, authentication).

                Present your findings clearly and concisely. You must use your available tools to research real-world technology choices.
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    cto_agent = create_openai_functions_agent(llm, tools, cto_prompt)
    return AgentExecutor(agent=cto_agent, tools=tools, verbose=True)