"""Test Google ADK structure for travel planner article."""

# Test imports
try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool
    print("ADK imports successful!")
    print(f"Agent class: {Agent}")
    print(f"FunctionTool class: {FunctionTool}")
    print("\nADK structure validated!")
except ImportError as e:
    print(f"Import error: {e}")
    # Try alternative import path
    try:
        from google.adk.agents.llm_agent import Agent
        print("Alternative import path works!")
    except ImportError as e2:
        print(f"Alternative also failed: {e2}")
