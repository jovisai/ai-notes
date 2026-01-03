"""Test OpenAI Swarm structure for multi-agent article."""

try:
    from swarm import Swarm, Agent
    print("Swarm imports successful!")
    print(f"Swarm: {Swarm}")
    print(f"Agent: {Agent}")

    # Create a simple agent (doesn't call API)
    def hello_function():
        return "Hello from function!"

    agent = Agent(
        name="TestAgent",
        instructions="You are a helpful assistant.",
        functions=[hello_function]
    )

    print(f"\nAgent created: {agent.name}")
    print(f"Functions: {[f.__name__ for f in agent.functions]}")
    print("Swarm structure validated!")

except ImportError as e:
    print(f"Import error: {e}")
