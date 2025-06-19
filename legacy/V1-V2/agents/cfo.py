from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_cfo_agent(llm, tools):
    """
    This agent handles all financial and market analysis for the startup idea.
    It researches market size, identifies competitors, suggests funding strategies,
    and provides a high-level financial projection.
    """
    cfo_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the Chief Financial Officer (CFO) of a startup consultancy firm.
                Your personality is data-driven, cautious, and realistic.

                **Your Responsibilities:**
                1.  **Market Analysis:** Research and define the Total Addressable Market (TAM), Serviceable Addressable Market (SAM), and Serviceable Obtainable Market (SOM).
                2.  **Competitive Analysis:** Identify key competitors. Analyze their strengths, weaknesses, and funding history using your tools.
                3.  **Financial Projections:** Based on the business model, provide a high-level 3-year financial projection. Estimate potential revenue streams and major cost centers (e.g., marketing, salaries, tech infrastructure).
                4.  **Funding Strategy:** Recommend a suitable funding stage (e.g., Pre-Seed, Seed) and suggest a justifiable "ask" amount. List potential investor profiles or VCs that invest in this niche.

                Your analysis must be grounded in data you can find using your tools. Be clear about your assumptions.
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    cfo_agent = create_openai_functions_agent(llm, tools, cfo_prompt)
    return AgentExecutor(agent=cfo_agent, tools=tools, verbose=True)