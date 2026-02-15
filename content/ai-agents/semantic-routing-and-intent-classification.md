---
title: "Semantic Routing and Intent Classification in AI Agent Systems"
date: 2025-11-09
tags: ["ai-agents", "semantic-routing", "intent-classification", "nlp", "embeddings"]
draft: false
summary: "Learn how semantic routing acts as an intelligent dispatcher for AI agents, directing user queries to the right tools, models, or workflows based on meaning rather than keywords."
---

## 1. Concept Introduction

### Simple Explanation

Imagine you walk into a massive hospital. Instead of wandering around trying to find the right department, there's a smart receptionist who understands your problem and immediately directs you to cardiology, orthopedics, or the pharmacy. **Semantic routing** is that receptionist for AI agent systems.

When a user asks your AI agent something, semantic routing determines *which tool, model, or workflow* should handle the request—not by matching keywords, but by understanding the **meaning** of the query.

### Technical Detail

Semantic routing is a classification mechanism that maps natural language inputs to predefined categories (routes) using vector similarity rather than rule-based pattern matching. Unlike traditional intent classification that relies on exact phrase matching or regex, semantic routing:

1. **Embeds** the user query into a high-dimensional vector space
2. **Compares** it against pre-computed route embeddings
3. **Selects** the route with highest similarity (typically cosine similarity)
4. **Dispatches** the query to the appropriate handler (tool, model, agent, or workflow)

This enables robust, fuzzy matching that handles paraphrasing, typos, and novel phrasings—critical for production agent systems where users never phrase things exactly as you expect.

## 2. Historical & Theoretical Context

### Origins

Intent classification has roots in **Natural Language Understanding (NLU)** research from the 1990s, particularly in dialog systems like MIT's JUPITER weather information system and AT&T's spoken dialog research. Early approaches used:

- **Rule-based pattern matching** (regex, keyword spotting)
- **Statistical classifiers** (Naive Bayes, SVM) trained on labeled examples
- **Slot-filling** for extracting parameters from recognized intents

The transition to **semantic routing** emerged from three key developments:

1. **Word2Vec (2013)**: Mikolov et al. showed that words could be embedded into vectors that capture semantic relationships
2. **Sentence Transformers (2019)**: Reimers and Gurevych developed efficient methods for embedding entire sentences with models like SBERT
3. **LLM Function Calling (2023)**: OpenAI's function calling and tool use capabilities showed the need for intelligent request routing in agentic systems

### Theoretical Foundation

Semantic routing builds on the **distributional hypothesis**: "Words that occur in similar contexts tend to have similar meanings" (Harris, 1954). Extended to sentences, this means queries with similar intent cluster in embedding space, even if they use different words.

This connects to **nearest-neighbor classification** in machine learning, where decision boundaries emerge from the geometry of the embedding space rather than explicit rules.

## 3. Algorithms & Math

### Core Algorithm

```
ALGORITHM: Semantic Routing

INPUT:
  - query: string (user input)
  - routes: list of Route objects, each containing:
      - name: string
      - description: string
      - examples: list of strings
      - embedding: vector (pre-computed)
  - threshold: float (minimum similarity score)
  - embedding_model: function that maps text → vector

OUTPUT: selected_route or None

STEPS:
1. query_embedding ← embedding_model(query)
2. similarities ← []
3. FOR each route IN routes:
     similarity ← cosine_similarity(query_embedding, route.embedding)
     similarities.append((route, similarity))
4. best_route, best_score ← max(similarities, key=score)
5. IF best_score >= threshold:
     RETURN best_route
   ELSE:
     RETURN None  // Fallback to default handler
```

### Mathematical Foundation

**Cosine Similarity** is the standard metric:

```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
                 = Σ(Aᵢ × Bᵢ) / (√Σ(Aᵢ²) × √Σ(Bᵢ²))
```

Where:
- `A · B` = dot product of vectors A and B
- `||A||` = Euclidean norm (magnitude) of vector A

Cosine similarity ranges from -1 to 1:
- **1**: Perfect alignment (identical direction)
- **0**: Orthogonal (completely unrelated)
- **-1**: Opposite direction

For semantic routing, we typically see scores of 0.6–0.95 for correct matches and 0.3–0.6 for incorrect ones, with thresholds commonly set around 0.7–0.8.

## 4. Design Patterns & Architectures

### Pattern: Router → Handler

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Semantic Router │ (Embedding + Similarity)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│Tool A │ │Workflow B│
└───────┘ └──────────┘
```

### Integration with Agent Architectures

**1. Pre-LLM Filtering** (Efficiency Pattern)
```
Query → Router → [Simple: Direct Answer] or [Complex: Full Agent Loop]
```
Saves LLM calls for simple queries that can be handled by templates or retrieval.

**2. Tool Selection** (Agentic Pattern)
```
Query → Router → [Calculator Tool | Search Tool | Code Executor | ...]
```
Replaces or augments LLM-based tool selection (like OpenAI function calling).

**3. Multi-Agent Dispatch** (Orchestration Pattern)
```
Query → Router → [SQL Agent | Python Agent | Research Agent | ...]
```
Routes to specialized agents in a multi-agent system (like AutoGen or CrewAI).

**4. Model Selection** (Optimization Pattern)
```
Query → Router → [GPT-4 | Claude | Llama-70B | Llama-7B]
```
Balances cost vs. capability by routing simple queries to cheaper models.

## 5. Practical Application

### Basic Python Implementation

```python
from openai import OpenAI
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

client = OpenAI()

@dataclass
class Route:
    name: str
    description: str
    examples: List[str]
    embedding: Optional[np.ndarray] = None

def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embedding vector for text."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SemanticRouter:
    def __init__(self, routes: List[Route], threshold: float = 0.75):
        self.routes = routes
        self.threshold = threshold
        self._initialize_routes()

    def _initialize_routes(self):
        """Pre-compute route embeddings from examples."""
        for route in self.routes:
            # Combine description and examples for richer representation
            route_text = f"{route.description}\n" + "\n".join(route.examples)
            route.embedding = get_embedding(route_text)

    def route(self, query: str) -> Optional[str]:
        """Route query to best matching route."""
        query_embedding = get_embedding(query)

        best_route = None
        best_score = -1

        for route in self.routes:
            similarity = cosine_similarity(query_embedding, route.embedding)
            if similarity > best_score:
                best_score = similarity
                best_route = route

        if best_score >= self.threshold:
            print(f"Routed to '{best_route.name}' (score: {best_score:.3f})")
            return best_route.name
        else:
            print(f"No confident route (best score: {best_score:.3f})")
            return None

# Define routes
routes = [
    Route(
        name="calculate",
        description="Mathematical calculations and arithmetic",
        examples=[
            "What is 234 * 567?",
            "Calculate the square root of 144",
            "How much is 15% of 200?"
        ]
    ),
    Route(
        name="search_web",
        description="Current events, news, and information requiring web search",
        examples=[
            "What's the weather in Tokyo?",
            "Who won the election?",
            "Latest news about AI"
        ]
    ),
    Route(
        name="code_help",
        description="Programming questions, debugging, and code examples",
        examples=[
            "How do I reverse a string in Python?",
            "Debug this JavaScript error",
            "Write a function to sort a list"
        ]
    ),
    Route(
        name="general_chat",
        description="General conversation, greetings, and casual questions",
        examples=[
            "Hello, how are you?",
            "Tell me a joke",
            "What's your favorite color?"
        ]
    )
]

# Create router
router = SemanticRouter(routes, threshold=0.70)

# Test queries
test_queries = [
    "What is 2 raised to the power of 10?",  # Should → calculate
    "What's happening in Ukraine today?",     # Should → search_web
    "How do I implement binary search?",      # Should → code_help
    "Hey there!",                             # Should → general_chat
    "Explain quantum mechanics"               # Ambiguous - might fail threshold
]

for query in test_queries:
    print(f"\nQuery: {query}")
    result = router.route(query)
```

### Framework Integration: LangChain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# After routing, dispatch to appropriate chain
def create_calculator_chain():
    prompt = ChatPromptTemplate.from_template(
        "You are a calculator. Solve: {query}\nProvide only the answer."
    )
    return prompt | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

def create_search_chain():
    # Integrate with web search tool
    prompt = ChatPromptTemplate.from_template(
        "Search for: {query}\n[Web search results would go here]"
    )
    return prompt | ChatOpenAI(model="gpt-4") | StrOutputParser()

# Dispatch based on route
def handle_query(query: str):
    route = router.route(query)

    if route == "calculate":
        chain = create_calculator_chain()
        return chain.invoke({"query": query})
    elif route == "search_web":
        chain = create_search_chain()
        return chain.invoke({"query": query})
    # ... other routes
    else:
        # Fallback to general-purpose agent
        return ChatOpenAI(model="gpt-4").invoke(query)
```

## 6. Comparisons & Tradeoffs

### Semantic Routing vs. Alternatives

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Semantic Routing** | Fast (vector lookup), no training data needed, handles paraphrasing | Requires good examples, may fail on edge cases | Production systems with 5–20 routes |
| **LLM-based Classification** | Highly flexible, excellent accuracy | Slow, expensive (full LLM call), high latency | High-stakes decisions, complex intents |
| **Fine-tuned Classifier** | Very fast, highly accurate for known classes | Requires labeled training data, inflexible | High-volume systems with stable intents |
| **Keyword/Regex** | Blazing fast, deterministic | Brittle, fails on paraphrasing, maintenance nightmare | Legacy systems, strict command interfaces |

### Key Tradeoffs

**Threshold Selection**:
- **High (0.85+)**: Fewer false positives, more fallbacks to default handler
- **Low (0.60–0.70)**: Higher coverage, risk of misrouting

**Number of Routes**:
- **Few (3–7)**: Fast, easy to maintain, clear separation
- **Many (20+)**: Risk of overlap, harder to distinguish, slower

**Example Quality**:
- More examples per route → better coverage but slower initialization
- Diverse phrasings → better robustness

## 7. Latest Developments & Research

### Recent Advances (2023–2025)

**1. Semantic Router Libraries**
- **Aurelio AI's Semantic Router** (2023): Production-ready library with Pinecone/Qdrant integration
- **LlamaIndex's Router Modules** (2024): Integrated routing for RAG applications

**2. Dynamic Route Learning**
The paper *"Adaptive Intent Classification via Online Clustering"* (Chen et al., 2024) showed how routes can be learned dynamically from user interactions, updating embeddings as new phrasings emerge.

**3. Hybrid Approaches**
*"CoRouter: Combining Semantic and LLM-based Routing"* (Liu et al., 2024) demonstrated a two-stage approach:
1. Semantic router for 80% of queries (fast path)
2. LLM fallback for ambiguous cases (slow path)

This achieved 94% accuracy while reducing inference costs by 60%.

**4. Multi-Vector Routing**
Instead of single embeddings, routes represented by **multiple prototype vectors** capturing different aspects of intent. Showed 15% improvement on the CLINC150 intent dataset.

### Benchmarks

**CLINC150**: 150-class intent classification dataset
- Classical ML (SVM): 87% accuracy
- Semantic routing (base): 91% accuracy
- Fine-tuned BERT: 97% accuracy
- GPT-4 classification: 96% accuracy (but 50x slower/more expensive)

### Open Problems

1. **Cold Start**: How to define routes with minimal examples?
2. **Drift Detection**: When do user intents shift requiring route updates?
3. **Explainability**: How to explain why a query was routed incorrectly?
4. **Multilingual Routing**: Handling mixed-language queries efficiently

## 8. Cross-Disciplinary Insight

### Connection to Network Routing

Semantic routing mirrors **packet routing** in computer networks:

| Network Routing | Semantic Routing |
|-----------------|------------------|
| IP address → destination network | Query embedding → intent category |
| Routing table | Route embeddings |
| Longest prefix match | Highest similarity score |
| BGP updates | Route embedding updates |
| Failover to default gateway | Fallback to general handler |

Just as **Software-Defined Networking (SDN)** made network routing programmable and adaptive, semantic routing makes agent dispatch programmable based on *meaning* rather than static rules.

### Neuroscience Parallel

The brain's **thalamus** acts as a semantic router, directing sensory inputs to appropriate cortical regions:
- Visual input → occipital lobe
- Sound → temporal lobe
- Touch → parietal lobe

Similarly, semantic routers direct different types of queries to specialized "processing regions" (tools/agents). Both use learned patterns rather than hardcoded rules, enabling flexible, context-dependent routing.

## 9. Daily Challenge

### Thought Exercise: Design a Router

**Scenario**: You're building a customer service agent that needs to handle:
- Billing questions
- Technical support
- Product recommendations
- General inquiries

**Task** (30 minutes):

1. **Define 4 routes** with descriptions and 3 examples each
2. **Identify edge cases**: Write 3 queries that might be ambiguous between routes
3. **Implement basic router** (use the code above as template)
4. **Test**: Run 10 sample queries and check accuracy

**Bonus**: What threshold would you use? How would you handle queries below the threshold?

### Coding Challenge

Extend the basic router to support **hierarchical routing**:

```python
# First route: Customer service → [Billing | Support | Sales]
# Then sub-route: Support → [Login Issues | Feature Questions | Bugs]

class HierarchicalRouter:
    def __init__(self, routes: dict):
        # routes = {
        #   "customer_service": SemanticRouter(...),
        #   "billing": SemanticRouter(...),
        #   ...
        # }
        pass

    def route(self, query: str) -> List[str]:
        # Return path like ["customer_service", "support", "login_issues"]
        pass
```

Can you make it work in under 50 lines of code?

## 10. References & Further Reading

### Papers

1. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**
   Reimers & Gurevych (2019)
   https://arxiv.org/abs/1908.10084

2. **Learning to Route in Similarity Graphs**
   Baranchuk et al. (2019)
   https://arxiv.org/abs/1905.10987

3. **Intent Classification and Slot Filling for Privacy Policies**
   Harkous et al. (2018)
   https://arxiv.org/abs/1805.00973

### Libraries & Tools

1. **Semantic Router by Aurelio AI**
   https://github.com/aurelio-labs/semantic-router
   Production-ready semantic routing with caching

2. **LlamaIndex Routers**
   https://docs.llamaindex.ai/en/stable/module_guides/querying/router/
   Query routers for RAG applications

3. **Embedchain**
   https://github.com/embedchain/embedchain
   RAG framework with built-in semantic routing

### Blog Posts

1. **"Building a Semantic Router"** by Aurelio Labs
   https://www.aurelio.ai/blog/semantic-router

2. **"Intent Classification with Sentence Transformers"**
   https://www.sbert.net/examples/applications/computing-embeddings/README.html

3. **"Routing Queries in Production LLM Applications"** by LangChain
   https://blog.langchain.dev/routing-queries/

### Datasets for Experimentation

1. **CLINC150**: 150 intent classes, 10 domains
   https://github.com/clinc/oos-eval

2. **ATIS (Airline Travel Information System)**
   Classic intent classification benchmark

3. **BANKING77**: 77 fine-grained banking intents
   https://github.com/PolyAI-LDN/task-specific-datasets

---

**Key Takeaway**: Semantic routing is the unsung hero of production AI agent systems. While flashier techniques like ReAct and chain-of-thought get more attention, semantic routing quietly ensures your expensive LLM calls are only used when necessary, routing simple queries to fast paths and complex ones to specialized handlers. Master this, and you'll build faster, cheaper, and more reliable agent systems.

**Next Steps**:
- Implement the basic router with your own routes
- Experiment with different thresholds
- Try hierarchical routing for complex domains
- Profile the latency impact in a real application

Now go build something intelligent! 🚀
