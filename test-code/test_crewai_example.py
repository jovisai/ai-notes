"""Test CrewAI implementation for comparison article."""
from crewai import Agent, Task, Crew

# This is a structural test - we verify the code compiles and structures are correct
# Actual LLM calls would require API keys

def create_crewai_research_crew():
    """Create a research crew with CrewAI."""

    # Define agents
    researcher = Agent(
        role="Research Analyst",
        goal="Find accurate information about the given topic",
        backstory="You are an expert researcher who finds reliable sources and extracts key insights.",
        verbose=True
    )

    writer = Agent(
        role="Content Writer",
        goal="Create clear, engaging summaries from research",
        backstory="You are a skilled writer who transforms complex information into readable content.",
        verbose=True
    )

    # Define tasks
    research_task = Task(
        description="Research the topic: {topic}. Find 3 key facts.",
        expected_output="A list of 3 key facts with sources.",
        agent=researcher
    )

    writing_task = Task(
        description="Write a summary based on the research findings.",
        expected_output="A 100-word summary of the key facts.",
        agent=writer,
        context=[research_task]
    )

    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    return crew


if __name__ == "__main__":
    print("CrewAI Structure Test")
    print("=" * 40)

    crew = create_crewai_research_crew()

    print(f"Agents: {[a.role for a in crew.agents]}")
    print(f"Tasks: {len(crew.tasks)}")
    print(f"Task dependencies: {crew.tasks[1].context is not None}")
    print("\nCrewAI structure validated successfully!")
