"""Test Agentic RAG implementation with LangGraph."""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator


# Define state for Agentic RAG
class RAGState(TypedDict):
    query: str
    search_queries: Annotated[list[str], operator.add]
    documents: Annotated[list[dict], operator.add]
    answer: str
    needs_more_info: bool
    iteration: int


# Simulated vector store
MOCK_DOCUMENTS = [
    {"id": 1, "content": "Python was created by Guido van Rossum in 1991.", "topic": "python history"},
    {"id": 2, "content": "Python uses indentation for code blocks.", "topic": "python syntax"},
    {"id": 3, "content": "LangGraph is built on top of LangChain.", "topic": "langgraph"},
    {"id": 4, "content": "RAG combines retrieval with generation.", "topic": "rag"},
    {"id": 5, "content": "Agentic RAG uses LLMs to decide retrieval strategy.", "topic": "agentic rag"},
]


def query_analyzer(state: RAGState) -> dict:
    """Analyze query and generate search queries."""
    query = state["query"]
    # Simulate LLM breaking down the query
    # In production: LLM would analyze and decompose the query
    search_queries = [query]  # Simple: just use original query

    if "compare" in query.lower() or "vs" in query.lower():
        # If comparing, search for both terms
        words = query.lower().replace("vs", " ").replace("compare", " ").split()
        search_queries = [w for w in words if len(w) > 3]

    return {"search_queries": search_queries}


def retriever(state: RAGState) -> dict:
    """Retrieve relevant documents based on search queries."""
    search_queries = state["search_queries"]

    retrieved = []
    for sq in search_queries:
        for doc in MOCK_DOCUMENTS:
            if sq.lower() in doc["content"].lower() or sq.lower() in doc["topic"]:
                if doc not in retrieved:
                    retrieved.append(doc)

    return {"documents": retrieved}


def relevance_checker(state: RAGState) -> dict:
    """Check if retrieved documents are sufficient."""
    docs = state["documents"]
    query = state["query"]

    # Simple heuristic: need at least 2 documents
    needs_more = len(docs) < 2 and state["iteration"] < 2

    return {"needs_more_info": needs_more, "iteration": state["iteration"] + 1}


def query_rewriter(state: RAGState) -> dict:
    """Rewrite query if more information is needed."""
    # Simulate query expansion
    original = state["query"]
    expanded = f"{original} definition examples"
    return {"search_queries": [expanded]}


def generator(state: RAGState) -> dict:
    """Generate answer from retrieved documents."""
    docs = state["documents"]
    query = state["query"]

    if not docs:
        answer = "I couldn't find relevant information to answer your question."
    else:
        # Simulate LLM generation
        context = " ".join([d["content"] for d in docs])
        answer = f"Based on {len(docs)} sources: {context[:200]}..."

    return {"answer": answer}


def should_continue(state: RAGState) -> Literal["rewrite", "generate"]:
    """Decide whether to rewrite query or generate answer."""
    if state["needs_more_info"]:
        return "rewrite"
    return "generate"


def create_agentic_rag_graph():
    """Build the Agentic RAG graph."""
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("analyze", query_analyzer)
    workflow.add_node("retrieve", retriever)
    workflow.add_node("check", relevance_checker)
    workflow.add_node("rewrite", query_rewriter)
    workflow.add_node("generate", generator)

    # Define flow
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "check")

    # Conditional: either rewrite or generate
    workflow.add_conditional_edges(
        "check",
        should_continue,
        {"rewrite": "rewrite", "generate": "generate"}
    )

    workflow.add_edge("rewrite", "retrieve")  # Loop back
    workflow.add_edge("generate", END)

    return workflow.compile()


if __name__ == "__main__":
    print("Agentic RAG Test")
    print("=" * 40)

    graph = create_agentic_rag_graph()

    # Test with a query
    result = graph.invoke({
        "query": "What is agentic RAG?",
        "search_queries": [],
        "documents": [],
        "answer": "",
        "needs_more_info": False,
        "iteration": 0
    })

    print(f"Query: {result['query']}")
    print(f"Documents found: {len(result['documents'])}")
    print(f"Iterations: {result['iteration']}")
    print(f"Answer: {result['answer']}")
    print("\nAgentic RAG validated successfully!")
