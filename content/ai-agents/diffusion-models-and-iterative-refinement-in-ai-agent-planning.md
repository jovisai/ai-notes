---
title: "Diffusion Models and Iterative Refinement in AI Agent Planning"
date: 2025-12-03
draft: false
tags: ["ai-agents", "planning", "diffusion-models", "generative-ai", "trajectory-optimization"]
description: "Explore how diffusion models enable AI agents to generate and refine complex action plans through iterative denoising, revolutionizing long-horizon planning and decision-making."
---

When a skilled chess player considers their next move, they don't immediately jump to the perfect solution. Instead, they start with a rough idea and progressively refine it—adjusting, correcting, and polishing until a strong strategy emerges. This iterative refinement process has now found a powerful mathematical formulation in AI: **diffusion models**.

Originally developed for image generation (think DALL-E, Stable Diffusion), diffusion models are revolutionizing how AI agents plan complex action sequences. Instead of generating a plan in one shot, agents can now start with random noise and gradually "denoise" it into a coherent, high-quality trajectory—enabling unprecedented flexibility in long-horizon planning.

## 1. Concept Introduction

### Simple Terms

Imagine you're sculpting with clay. You don't instantly create a perfect statue—you start with a rough blob and progressively refine it. You smooth rough edges, adjust proportions, and add details in multiple passes.

**Diffusion planning** works similarly: an agent starts with pure randomness (like a blob of clay) and iteratively refines it into a concrete action plan. At each refinement step, the agent asks: "Which part of this plan looks noisy or unrealistic?" and smooths it out, eventually converging to a valid, high-quality trajectory.

The key insight: **planning as iterative denoising** rather than one-shot generation.

### Technical Detail

**Diffusion models** for planning treat action sequences as high-dimensional vectors that can be progressively refined through a learned denoising process.

**Core mechanism:**
1. **Forward process (training)**: Gradually add Gaussian noise to ground-truth trajectories until they become pure noise
2. **Reverse process (inference)**: Learn a neural network that reverses this process—starting from noise and iteratively denoising to recover valid plans

**Mathematical formulation:**

```
Forward: x_t = √(α_t) x_0 + √(1-α_t) ε,  ε ~ N(0, I)
Reverse: x_{t-1} = μ_θ(x_t, t) + σ_t z,   z ~ N(0, I)
```

Where:
- `x_0` is a real trajectory (state-action sequence)
- `x_t` is the noisy version at timestep t
- `α_t` controls noise schedule (decreases from 1 to 0)
- `μ_θ` is the learned denoising function

**Key advantages:**
- **Multimodal planning**: Can represent multiple valid solutions simultaneously
- **Constraint satisfaction**: Easy to inject constraints during denoising
- **Robustness**: Graceful degradation under uncertainty
- **Long-horizon**: Better at capturing global structure than autoregressive methods

## 2. Historical & Theoretical Context

### Origins

**Diffusion models** emerged from multiple lineages:

1. **Non-equilibrium thermodynamics** (Sohl-Dickstein et al., 2015): Original diffusion probabilistic models drew inspiration from thermodynamic processes
2. **Score-based generative models** (Song & Ermon, 2019): Connected diffusion to score matching and Langevin dynamics
3. **Denoising diffusion probabilistic models (DDPM)** (Ho et al., 2020): Simplified training and sparked the generative AI revolution

### Application to Planning

The breakthrough for agent planning came in 2022-2023:

- **Diffuser** (Janner et al., 2022): First application to offline RL and trajectory optimization
- **Planning with Diffusion** (Ajay et al., 2022): Extended to vision-based robotic manipulation
- **AdaptDiffuser** (Liang et al., 2023): Added online replanning and constraint handling

**Theoretical connection**: Diffusion planning is a form of **trajectory optimization** that uses learned generative models instead of gradient-based optimization. It relates to:
- **Optimal control**: Solving for action sequences that maximize cumulative reward
- **Sampling-based planning**: Monte Carlo methods for exploring solution spaces
- **Energy-based models**: Treating valid plans as low-energy configurations

### Why It Matters Now

Traditional planning methods struggle with:
- **High dimensionality**: Exponential search spaces in long-horizon tasks
- **Multimodality**: Multiple equally valid solutions
- **Partial observability**: Uncertainty about world state

Diffusion models naturally handle all three through their generative, iterative refinement process.

## 3. Algorithms & Math

### Diffusion Planning Algorithm

**Training Phase:**

```
Input: Dataset D of trajectories τ = (s_0, a_0, s_1, a_1, ..., s_T, a_T)

For each trajectory τ in D:
  1. Sample timestep t ~ Uniform(1, T_diffusion)
  2. Sample noise ε ~ N(0, I)
  3. Create noisy trajectory: τ_t = √(α_t) τ + √(1-α_t) ε
  4. Predict noise: ε_pred = Network_θ(τ_t, t, context)
  5. Update θ to minimize: ||ε - ε_pred||²

Output: Trained denoising network θ*
```

**Planning Phase (Inference):**

```
Input: Current state s_current, goal g
Output: Planned trajectory τ*

1. Initialize: τ_T ~ N(0, I)  # Start with pure noise

2. For t = T down to 1:
     # Denoise one step
     ε_pred = Network_θ(τ_t, t, context=[s_current, g])
     τ_{t-1} = Denoise(τ_t, ε_pred, t)

     # Optional: Apply constraints (e.g., ensure first state = s_current)
     τ_{t-1} = ProjectConstraints(τ_{t-1}, s_current)

3. Return τ_0  # Fully denoised trajectory
```

### Denoising Function

The core denoising step uses the **DDPM reverse process**:

```python
def denoise_step(x_t, epsilon_pred, t, alpha_schedule):
    """Single step of reverse diffusion"""
    alpha_t = alpha_schedule[t]
    alpha_t_prev = alpha_schedule[t-1]

    # Predicted clean trajectory
    x_0_pred = (x_t - sqrt(1 - alpha_t) * epsilon_pred) / sqrt(alpha_t)

    # Reverse process mean
    mu = (sqrt(alpha_t_prev) * (1 - alpha_t / alpha_t_prev) * x_0_pred +
          sqrt(alpha_t) * (1 - alpha_t_prev) * x_t) / (1 - alpha_t)

    # Add noise (except final step)
    if t > 1:
        sigma = sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        return mu + sigma * torch.randn_like(x_t)
    return mu
```

### Constraint Injection

A major advantage: constraints can be enforced during generation.

**Hard constraints (projection):**
```python
def project_constraints(trajectory, current_state, goal_state):
    # Ensure trajectory starts at current state
    trajectory[0] = current_state

    # Ensure trajectory ends at goal
    trajectory[-1] = goal_state

    # Ensure actions are valid
    trajectory.actions = torch.clamp(trajectory.actions, action_min, action_max)

    return trajectory
```

**Soft constraints (guidance):**
```python
def guided_denoise(x_t, t, constraint_fn):
    # Unconditional denoising
    epsilon_pred = network(x_t, t)

    # Compute gradient of constraint satisfaction
    with torch.enable_grad():
        x_0_pred = predict_x0(x_t, epsilon_pred, t)
        loss = constraint_fn(x_0_pred)
        grad = torch.autograd.grad(loss, x_t)[0]

    # Adjust prediction toward satisfying constraints
    epsilon_adjusted = epsilon_pred - scale * grad
    return epsilon_adjusted
```

## 4. Design Patterns & Architectures

### Integration with Agent Architectures

Diffusion planning fits naturally into the **planner-executor-memory** loop:

```
┌─────────────────────────────────────────────┐
│          AGENT COGNITIVE LOOP               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐      ┌──────────────────┐   │
│  │ Perceive │─────>│ Diffusion Planner│   │
│  │  State   │      │                  │   │
│  └──────────┘      │ 1. Sample noise  │   │
│       │            │ 2. Denoise T steps│   │
│       │            │ 3. Return τ*     │   │
│       │            └────────┬─────────┘   │
│       │                     │              │
│  ┌────▼──────┐      ┌──────▼─────┐        │
│  │  Memory/  │      │  Executor  │        │
│  │ Context   │<─────│ (Low-level │        │
│  │           │      │  control)  │        │
│  └───────────┘      └──────┬─────┘        │
│       │                    │               │
│       │            ┌───────▼──────┐        │
│       └───────────>│   Re-plan?   │        │
│                    └──────────────┘        │
└─────────────────────────────────────────────┘
```

### Common Patterns

**1. Receding Horizon Diffusion Planning**

Don't execute the entire plan—only the first few steps, then replan:

```python
class RecedingHorizonDiffusionAgent:
    def __init__(self, diffusion_model, horizon=20, replan_freq=5):
        self.model = diffusion_model
        self.horizon = horizon
        self.replan_freq = replan_freq

    def act(self, state, goal):
        # Generate plan every replan_freq steps
        if self.steps % self.replan_freq == 0:
            self.plan = self.model.plan(state, goal, horizon=self.horizon)

        # Execute next action from plan
        action = self.plan.actions[self.steps % self.replan_freq]
        self.steps += 1
        return action
```

**2. Hierarchical Diffusion Planning**

High-level diffusion for waypoints, low-level for action details:

```
High-level diffuser: state_0 -> waypoint_1 -> waypoint_2 -> goal
                                    │              │
Low-level diffuser:         [detailed actions] [detailed actions]
```

**3. Test-Time Constraint Optimization**

Use classifier guidance to satisfy new constraints at inference time:

```python
def plan_with_safety_constraint(diffuser, state, goal, obstacle_map):
    def collision_cost(trajectory):
        """Penalize trajectories that hit obstacles"""
        positions = trajectory.states[:, :2]  # x, y positions
        return obstacle_map.query(positions).sum()

    return diffuser.guided_plan(
        state,
        goal,
        constraint_fn=collision_cost,
        guidance_scale=2.0
    )
```

## 5. Practical Application

### Example: Robotic Manipulation with Diffusion Planning

Let's build a simple diffusion planner for a robot arm:

```python
import torch
import torch.nn as nn
import numpy as np

class DiffusionPlanner(nn.Module):
    def __init__(self, state_dim, action_dim, horizon, hidden_dim=256):
        super().__init__()
        self.horizon = horizon
        self.traj_dim = (state_dim + action_dim) * horizon

        # Denoising network: MLP that takes noisy trajectory + timestep
        self.net = nn.Sequential(
            nn.Linear(self.traj_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.traj_dim)
        )

        # Noise schedule (linear)
        self.T = 100
        self.betas = torch.linspace(1e-4, 0.02, self.T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_t, t):
        """Predict noise at timestep t"""
        t_embed = t.float().unsqueeze(-1) / self.T
        x_input = torch.cat([x_t, t_embed], dim=-1)
        return self.net(x_input)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to trajectory"""
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar = self.alpha_bars[t]
        return (torch.sqrt(alpha_bar) * x_0 +
                torch.sqrt(1 - alpha_bar) * noise), noise

    def p_sample(self, x_t, t):
        """Reverse diffusion: denoise one step"""
        # Predict noise
        epsilon_pred = self(x_t, t)

        # Get schedule values
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]

        # Predicted x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * epsilon_pred) / torch.sqrt(alpha_bar)

        if t > 0:
            noise = torch.randn_like(x_t)
            alpha_bar_prev = self.alpha_bars[t-1]

            # Reverse process mean
            mu = (torch.sqrt(alpha_bar_prev) * (1 - alpha) * x_0_pred +
                  torch.sqrt(alpha) * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar)

            # Variance
            sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha))
            return mu + sigma * noise

        return x_0_pred

    def plan(self, current_state, goal, num_samples=1):
        """Generate a plan from current state to goal"""
        # Start with random noise
        x_T = torch.randn(num_samples, self.traj_dim)

        # Iteratively denoise
        x_t = x_T
        for t in reversed(range(self.T)):
            t_batch = torch.tensor([t] * num_samples)
            x_t = self.p_sample(x_t, t_batch)

            # Enforce constraints: first state = current, last state = goal
            # (This is a simplified projection)
            if t % 10 == 0:  # Apply every 10 steps for efficiency
                x_t[:, :len(current_state)] = torch.tensor(current_state)
                x_t[:, -len(goal):] = torch.tensor(goal)

        return x_t  # Returns denoised trajectory

# Training loop (simplified)
def train_diffusion_planner(planner, trajectories, epochs=100):
    optimizer = torch.optim.Adam(planner.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for traj_batch in trajectories:
            # Sample random timestep
            t = torch.randint(0, planner.T, (len(traj_batch),))

            # Add noise
            noisy_traj, noise = planner.q_sample(traj_batch, t)

            # Predict noise
            noise_pred = planner(noisy_traj, t)

            # MSE loss
            loss = ((noise - noise_pred) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Usage example
if __name__ == "__main__":
    state_dim = 7  # Robot joint angles
    action_dim = 7  # Joint velocities
    horizon = 20   # 20-step plans

    planner = DiffusionPlanner(state_dim, action_dim, horizon)

    # Assume we have collected trajectories (shape: [batch, traj_dim])
    # trajectories = load_robot_data()
    # train_diffusion_planner(planner, trajectories)

    # At test time: plan from current state to goal
    current_state = np.array([0.0, 0.5, 0.0, -1.2, 0.0, 1.5, 0.0])
    goal_state = np.array([0.5, 0.3, 0.2, -0.8, 0.1, 1.0, 0.3])

    planned_traj = planner.plan(current_state, goal_state)
    print(f"Generated trajectory shape: {planned_traj.shape}")
```

### Integration with LangGraph

For higher-level AI agent workflows, diffusion planning can be a specialized node:

```python
from langgraph.graph import StateGraph, END

class AgentState:
    current_state: dict
    goal: dict
    plan: list
    execution_step: int

def diffusion_plan_node(state: AgentState):
    """LangGraph node that uses diffusion planning"""
    planner = DiffusionPlanner(...)  # Load trained model

    # Convert state/goal to vector representation
    state_vec = state_to_vector(state.current_state)
    goal_vec = state_to_vector(state.goal)

    # Generate plan
    trajectory = planner.plan(state_vec, goal_vec)

    # Convert back to action sequence
    actions = trajectory_to_actions(trajectory)

    return {"plan": actions, "execution_step": 0}

def execute_action_node(state: AgentState):
    """Execute next action from plan"""
    action = state.plan[state.execution_step]
    # Execute in environment
    result = environment.step(action)

    return {
        "current_state": result.state,
        "execution_step": state.execution_step + 1
    }

def should_replan(state: AgentState):
    """Decide if replanning is needed"""
    if state.execution_step % 5 == 0:  # Replan every 5 steps
        return "plan"
    if reached_goal(state):
        return END
    return "execute"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("plan", diffusion_plan_node)
workflow.add_node("execute", execute_action_node)

workflow.set_entry_point("plan")
workflow.add_conditional_edges("execute", should_replan, {
    "plan": "plan",
    "execute": "execute",
    END: END
})
workflow.add_edge("plan", "execute")

agent = workflow.compile()
```

## 6. Comparisons & Tradeoffs

### Diffusion Planning vs. Alternatives

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Diffusion Planning** | Multimodal, handles constraints, long-horizon, one-shot generation | Slow inference (many denoising steps), requires offline data, stochastic |
| **Autoregressive (GPT-style)** | Fast inference, flexible, pre-trained models available | Struggles with long-horizon, error accumulation, hard to inject constraints |
| **Optimization (CEM, iLQG)** | Guarantees local optimality, precise | Requires differentiable models, local minima, slow for high-dim |
| **Sampling (RRT, PRM)** | No learning needed, completeness guarantees | Curse of dimensionality, ignores learned patterns |
| **Value-based RL (Q-learning)** | Online learning, sample efficient | Discrete actions, needs value function, myopic |

### When to Use Diffusion Planning

**Best for:**
- Long-horizon manipulation/navigation tasks (20-100 steps)
- Problems with multiple valid solutions
- Offline learning from demonstration data
- Tasks requiring constraint satisfaction (safety, physics)
- Continuous, high-dimensional action spaces

**Avoid for:**
- Real-time systems requiring <10ms responses
- Sparse data regimes (needs lots of trajectories)
- Purely discrete action spaces
- Tasks where single optimal solution is known

### Computational Tradeoffs

**Training**: O(N × T) where N = dataset size, T = diffusion steps (~100)
- Similar cost to training other generative models
- Can leverage GPU parallelism effectively

**Inference**: O(T × H) where T = denoising steps, H = horizon
- Typical: 100 denoising steps × 20 step horizon = 2000 forward passes
- **Speedup techniques**:
  - DDIM sampling (Song et al., 2021): Reduce T to 10-20 steps
  - Distillation: Train student model to match in 1 step
  - Progressive distillation: Iteratively halve steps

```python
# Fast sampling with DDIM
def ddim_sample(self, x_T, steps=20):
    """Much faster than DDPM (100 steps)"""
    timesteps = torch.linspace(0, self.T-1, steps).long()

    x_t = x_T
    for i in reversed(range(steps)):
        t = timesteps[i]
        epsilon_pred = self(x_t, t)

        # DDIM update (deterministic)
        alpha_bar = self.alpha_bars[t]
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * epsilon_pred) / torch.sqrt(alpha_bar)

        if i > 0:
            alpha_bar_prev = self.alpha_bars[timesteps[i-1]]
            x_t = (torch.sqrt(alpha_bar_prev) * x_0_pred +
                   torch.sqrt(1 - alpha_bar_prev) * epsilon_pred)
        else:
            x_t = x_0_pred

    return x_t
```

## 7. Latest Developments & Research

### Recent Breakthroughs (2023-2025)

**1. Decision Diffuser (Ajay et al., 2023)**
- Unified framework for decision-making as conditional generation
- State-of-the-art on D4RL offline RL benchmarks
- Key insight: Treat entire decision trajectory as single generative modeling problem

**2. Diffusion-QL (Wang et al., 2023)**
- Combines Q-learning with diffusion for online RL
- Uses diffusion to generate candidate actions, Q-function to select best
- Achieves online learning without sacrificing multimodal expressiveness

**3. MotionDiffuse (Zhang et al., 2024)**
- Human motion generation for robotics
- Incorporates physical constraints during denoising
- Enables zero-shot transfer to new environments

**4. SayPlan (Rana et al., 2024)**
- Large language models guide diffusion planning
- LLM provides semantic waypoints, diffusion fills in detailed trajectories
- Bridges symbolic reasoning with continuous control

**5. Diffusion Forcing (Chen et al., 2025)**
- Training technique that improves long-horizon generation
- Forces model to denoise from arbitrary partial trajectories
- Reduces compounding errors in autoregressive-style execution

### Current Benchmarks

**D4RL (Deep Data-Driven RL)**: Standard offline RL suite
- Diffuser achieves 80%+ normalized score on locomotion tasks
- Outperforms value-based methods on maze navigation

**CLEVRRobot**: Vision-based manipulation
- Diffusion planning: 73% success rate
- Autoregressive baselines: 45-60%

**Metrics tracked**:
- Success rate on held-out goals
- Constraint violation rate
- Inference time per plan
- Sample efficiency during training

### Open Problems

1. **Sample efficiency**: Still requires 100K+ demonstration trajectories
2. **Extrapolation**: Performance degrades on out-of-distribution states
3. **Interpretability**: Hard to debug why a particular plan was generated
4. **Partial observability**: Most work assumes full state access
5. **Hierarchical planning**: Scaling to 1000+ step tasks remains challenging

### Research Frontiers

- **Diffusion + Foundation Models**: Using pre-trained vision/language models as diffusion backbones
- **Multi-agent diffusion**: Coordinating plans across agents
- **Causal diffusion**: Incorporating causal structure into generation
- **Safe exploration**: Using diffusion to generate safe exploration trajectories

## 8. Cross-Disciplinary Insight

Diffusion planning connects to several fields:

### Statistical Physics

The forward/reverse diffusion process mirrors **thermodynamic equilibration**:
- Forward process = increasing entropy (order → chaos)
- Reverse process = decreasing entropy (chaos → order)
- Learning = finding the path through phase space

This connects to **Langevin dynamics** and **stochastic differential equations** (SDEs).

### Neuroscience

Human motor planning exhibits similar **iterative refinement**:
- Initial rough motor plan in premotor cortex
- Progressive refinement through basal ganglia loops
- Online corrections via cerebellar feedback

**Predictive coding**: The brain constantly predicts sensory input and corrects errors—analogous to diffusion's denoising.

### Systems Theory

Diffusion planning is a form of **receding horizon control** (Model Predictive Control):
- Generate trajectory over horizon
- Execute first few steps
- Replan with updated state

The stochastic nature provides natural **exploration** in uncertain environments.

### Economics

The noise schedule resembles **simulated annealing** in optimization:
- High initial noise = broad exploration (high temperature)
- Progressive noise reduction = exploitation (cooling)
- Avoids local optima through stochastic jumps

## 9. Daily Challenge

### Thought Exercise

**Question**: Why might diffusion planning outperform autoregressive action generation (like GPT-style models) for long-horizon robot manipulation?

*Hint: Think about error accumulation and multimodality.*

<details>
<summary>Solution</summary>

**Key insights:**

1. **No error accumulation**: Autoregressive models generate actions sequentially: a₀ → a₁ → a₂. Errors early in the sequence compound, leading to drift. Diffusion generates the entire trajectory jointly, so errors in one part don't cascade.

2. **Multimodal solutions**: Robot tasks often have multiple valid solutions (e.g., "pick up the cup"—grasp handle or sides?). Autoregressive models must commit to one mode early. Diffusion naturally represents multimodal distributions until the final denoising steps.

3. **Global coherence**: Diffusion considers the entire trajectory at once, ensuring actions are globally consistent. Autoregressive models are myopic—each action only conditions on the past, not the future goal.

4. **Constraint satisfaction**: Hard constraints (e.g., "don't hit the table") are easier to enforce in diffusion—you can project the entire trajectory. In autoregressive models, you might violate constraints partway through.

</details>

### Coding Exercise (30 minutes)

Implement a **1D navigation task** with obstacles using simplified diffusion planning.

**Setup:**
- Agent starts at position 0, goal at position 10
- Obstacles at positions [3, 7]
- Plan a trajectory (sequence of positions) that avoids obstacles

**Task:**
1. Implement simplified forward/reverse diffusion (5 timesteps only)
2. Add constraint: trajectory must not pass through obstacles
3. Visualize the denoising process

**Starter code:**

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleDiffusionPlanner:
    def __init__(self, horizon=10, diffusion_steps=5):
        self.horizon = horizon
        self.T = diffusion_steps
        # Simple linear noise schedule
        self.betas = np.linspace(0.01, 0.3, self.T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def forward_diffusion(self, trajectory):
        """Add noise to clean trajectory"""
        # TODO: Implement forward process
        pass

    def reverse_diffusion(self, noisy_traj, timestep):
        """Remove noise (simplified: just average with neighbors)"""
        # TODO: Implement simple denoising
        # Hint: Use smoothing + random walk bias toward goal
        pass

    def project_constraints(self, traj, obstacles):
        """Push trajectory away from obstacles"""
        # TODO: If any position is near obstacle, nudge it away
        pass

    def plan(self, start, goal, obstacles):
        """Generate plan via iterative denoising"""
        # Start with random trajectory
        traj = np.random.randn(self.horizon)

        # Denoise
        for t in reversed(range(self.T)):
            traj = self.reverse_diffusion(traj, t)
            traj = self.project_constraints(traj, obstacles)

            # Enforce boundary conditions
            traj[0] = start
            traj[-1] = goal

        return traj

# Test
planner = SimpleDiffusionPlanner()
trajectory = planner.plan(start=0, goal=10, obstacles=[3, 7])

# Visualize
plt.plot(trajectory, marker='o')
plt.axhline(y=3, color='r', linestyle='--', label='Obstacles')
plt.axhline(y=7, color='r', linestyle='--')
plt.xlabel('Timestep')
plt.ylabel('Position')
plt.title('Diffusion-Generated Trajectory')
plt.legend()
plt.show()
```

**Bonus**: Try visualizing the trajectory at each denoising step to see how it progressively clarifies.

## 10. References & Further Reading

### Foundational Papers

1. **Denoising Diffusion Probabilistic Models (DDPM)**
   Ho et al., NeurIPS 2020
   https://arxiv.org/abs/2006.11239

2. **Diffuser: Planning with Diffusion for Flexible Behavior Synthesis**
   Janner et al., ICML 2022
   https://arxiv.org/abs/2205.09991
   Code: https://github.com/jannerm/diffuser

3. **Planning with Diffusion for Flexible Behavior Synthesis**
   Ajay et al., CoRL 2022
   https://arxiv.org/abs/2205.09991

### Recent Advances

4. **Is Conditional Generative Modeling all you need for Decision-Making?**
   Ajay et al., ICLR 2023
   https://arxiv.org/abs/2211.15657

5. **Diffusion-QL: Diffusion Models for Q-Learning**
   Wang et al., NeurIPS 2023
   https://arxiv.org/abs/2208.06193

6. **Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion**
   Chen et al., 2024
   https://arxiv.org/abs/2407.01392

### Practical Resources

7. **Hugging Face Diffusers Library**
   https://github.com/huggingface/diffusers
   (Not planning-specific, but excellent diffusion implementation reference)

8. **Berkeley Robot Learning Lab**
   https://rail.eecs.berkeley.edu/
   (Multiple papers on diffusion for robotics)

9. **Lil'Log: Diffusion Models Tutorial**
   https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
   (Excellent mathematical overview)

### Implementations

10. **Diffusion Policy (Robotic Manipulation)**
    Chi et al., RSS 2023
    https://github.com/columbia-ai-robotics/diffusion_policy

11. **MotionDiffuse (Human Motion Generation)**
    https://github.com/mingyuan-zhang/MotionDiffuse

### Blog Posts

12. **The Annotated Diffusion Model**
    https://huggingface.co/blog/annotated-diffusion
    (Step-by-step implementation walkthrough)

13. **Diffusion Models from Scratch**
    https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/

---

Diffusion models represent a paradigm shift in AI agent planning: from brittle one-shot generation to robust iterative refinement. As these methods mature, expect them to become standard in robotics, game AI, and autonomous systems—anywhere agents need to generate complex, constrained action sequences in uncertain environments. The future of planning is probabilistic, iterative, and gracefully handles the messiness of the real world.
