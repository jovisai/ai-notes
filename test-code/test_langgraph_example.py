"""Test LangGraph implementation for comparison article."""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

# Define state
class ResearchState(TypedDict):
    topic: str
    research: Annotated[list[str], operator.add]
    summary: str


def research_node(state: ResearchState) -> dict:
    """Simulate research step."""
    topic = state["topic"]
    # In production, this would call an LLM
    facts = [
        f"Fact 1 about {topic}",
        f"Fact 2 about {topic}",
        f"Fact 3 about {topic}"
    ]
    return {"research": facts}


def writing_node(state: ResearchState) -> dict:
    """Simulate writing step."""
    research = state["research"]
    # In production, this would call an LLM
    summary = f"Summary of {len(research)} research findings about {state['topic']}."
    return {"summary": summary}


def create_langgraph_research_workflow():
    """Create a research workflow with LangGraph."""

    # Build the graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", writing_node)

    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    # Compile
    return workflow.compile()


if __name__ == "__main__":
    print("LangGraph Structure Test")
    print("=" * 40)

    graph = create_langgraph_research_workflow()

    # Test execution
    result = graph.invoke({
        "topic": "artificial intelligence",
        "research": [],
        "summary": ""
    })

    print(f"Topic: {result['topic']}")
    print(f"Research items: {len(result['research'])}")
    print(f"Summary: {result['summary']}")
    print("\nLangGraph structure validated successfully!")
