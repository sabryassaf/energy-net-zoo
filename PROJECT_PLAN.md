# Project Plan: Robust and Safe RL ISO Agent for Smart Grid Control

## 1. Project Explanation

### 1.1. Background
Modern electricity grids are evolving towards decentralized networks rich in renewable energy sources. This shift necessitates dynamic, real-time control of electricity pricing, for which intelligent agents offer a promising solution. Reinforcement Learning (RL) can enable agents to adaptively adjust prices to balance supply and demand. However, a critical challenge is ensuring grid stability, as uncertain demand and potentially unsafe actions from RL agents can lead to instability (e.g., blackouts, equipment damage).

### 1.2. Goals
The primary goal of this project is to develop a **Robust and Safe Reinforcement Learning (RL) agent for an Independent System Operator (ISO)**. This agent must:
1.  **Maintain supply-demand balance:** Effectively manage electricity pricing to ensure supply meets demand, even when faced with noisy or uncertain observations of demand.
2.  **Adhere to safety constraints:** Never breach hard safety limits critical to grid operation, such as voltage and frequency stability, and prevent damage to components like batteries (e.g., over-voltage, over-current).
3.  **Be stress-tested:** Successfully pass a suite of rigorous tests designed to evaluate its performance under adversarial conditions, including unusual demand patterns, generator outages, and sensor noise.

### 1.3. Motivation
-   **Economic Impact:** Preventing grid failures like blackouts and equipment damage can save millions in operational and repair costs.
-   **Addressing Uncertainty:** As grids integrate more renewables (solar, wind) and distributed storage, the inherent uncertainty in supply and demand skyrockets, making intelligent and safe control paramount.
-   **Grid Modernization:** Power systems are transitioning from centralized generation to highly decentralized architectures, requiring new control paradigms.

## 2. Development Steps

### Step 1: Environment Setup & Familiarization
-   **Install OmniSafe:** Add `omnisafe` to the project dependencies (`rl-baselines3-zoo/requirements.txt`) and install it into the Python environment.
    -   *Status: Dependency added to `requirements.txt`. Needs installation.*
-   **Understand `energy_net` Package:**
    -   Thoroughly review the API and functionality of the externally-installed `energy_net` package. This is the core simulation environment.
    -   Identify how to:
        -   Define/configure the ISO agent and its interaction with the grid.
        -   Access relevant observations (demand, grid state, sensor readings).
        -   Define the action space (pricing decisions).
        -   Obtain reward signals for supply-demand balance.
        -   Extract data relevant to safety constraints (voltage, frequency, battery states).
        -   Inject noise into demand signals and sensor readings.
        -   Simulate generator outages and other fault conditions.
-   **Familiarize with `rl-baselines3-zoo`:** Understand the existing training scripts (`train.py`, `exp_manager.py`), configuration, and workflows within the `rl-baselines3-zoo` directory, as this will be adapted.

### Step 2: Safe RL Agent Implementation (ISO Agent)
-   **Integrate OmniSafe with Training Loop:**
    -   Modify the training scripts in `rl-baselines3-zoo` (or create new ones) to use OmniSafe's trainers and agent wrappers.
    -   Ensure compatibility with the `energy_net` environment, potentially by creating custom wrappers if OmniSafe's defaults for `Safety-Gymnasium` are not directly applicable.
-   **Define Safety Constraints:**
    -   Formally translate the physical safety limits (voltage, frequency, battery parameters) into cost functions or constraints that OmniSafe algorithms can interpret.
    -   This will involve mapping specific states/observations from the `energy_net` environment to these cost signals.
-   **Implement Selected Safe RL Algorithms:**
    -   Start with one or two algorithms from the list (e.g., PPO-Lagrangian, CPO) and progressively add others.
    -   Configure and tune hyperparameters for these algorithms within the context of the `energy_net` ISO environment.

### Step 3: Robustness Enhancement
-   **Domain Randomization:**
    -   Develop mechanisms to introduce variability into the training environment:
        -   Noisy demand profiles.
        -   Noisy sensor measurements.
    -   Train the ISO agent across these randomized conditions to improve its generalization.
-   **Adversarial Training (Initial Phase):**
    -   Design and implement methods to simulate worst-case input perturbations or scenarios during training. This could involve:
        -   Rule-based generation of challenging demand spikes or dips.
        -   Simulating specific sequences of generator outages.

### Step 4: Stress Test Suite Development
-   **Define Adversarial Scenarios:**
    -   Formalize a set of challenging test cases beyond standard training variations:
        -   Extreme demand fluctuations (sustained peaks, sudden drops).
        -   Cascading generator outages.
        -   Significant sensor noise or sensor failures (if modellable).
        -   Combinations of the above.
-   **Implement Test Harness:**
    -   Create scripts or a framework to systematically run the trained ISO agent against these stress test scenarios.
    -   This harness should automatically collect all relevant metrics.

### Step 5: Evaluation and Iteration
-   **Define Detailed Metrics:**
    -   **Safety Violations:** Frequency, duration, and magnitude of breaching defined hard limits (voltage, frequency, battery current/voltage).
    -   **Reliability:**
        -   Supply/Demand Mismatch (e.g., unserved energy, curtailment).
        -   Frequency of blackouts or brownouts.
    -   **Economic Efficiency:** Dispatch cost, pricing stability.
-   **Execute Evaluation Pipeline:**
    -   **Individual Benchmarking:** Evaluate each trained Safe RL ISO agent variant against nominal and adversarial scenarios.
    -   **(Optional/Future) Hybrid Agent Combinations:** If other agent types (e.g., from `energy_net`) are involved, test combinations.
    -   **(Optional/Future) Multi-agent Validation:** Specifically test the ISO agent in conjunction with PCS (Price-Controlled Storage) agents, as hinted by the existing codebase.
-   **Analyze Results and Iterate:** Based on performance, identify weaknesses and iterate on:
    -   Algorithm choice and tuning.
    -   Constraint definitions.
    -   Robustness techniques.
    -   Environment model fidelity (if adjustments to `energy_net` are possible or needed).

## 3. Benchmarks and Evaluation Plan

### 3.1. Evaluation Metrics
-   **Primary Safety Metrics:**
    -   Number of timesteps where voltage limits are violated.
    -   Number of timesteps where frequency limits are violated.
    -   Number of timesteps where battery over-voltage/over-current occurs.
    -   Severity of violations (e.g., max deviation from limits).
-   **Reliability Metrics:**
    -   Total unserved energy (blackouts).
    -   Frequency and duration of periods with supply-demand mismatch.
    -   Reserve shortage instances.
-   **Economic Metrics:**
    -   Total operational cost (e.g., generation cost, balancing cost).
    -   Price volatility.
-   **Algorithm-Specific Metrics (from OmniSafe):**
    -   Average reward per episode.
    -   Average cost (constraint violation) per episode.

### 3.2. Test Scenarios
1.  **Nominal Conditions:**
    -   Standard demand profiles (e.g., daily, weekly patterns without extreme events).
    -   No unexpected generator outages.
    -   Low sensor noise.
2.  **Domain Randomization Scenarios (during training & testing):**
    -   Varied demand profiles based on historical data + synthetic noise.
    -   Different levels of sensor noise applied to agent observations.
3.  **Adversarial / Stress Test Scenarios:**
    -   **High Demand Peaks:** Sudden, unexpected spikes in electricity demand.
    -   **Sudden Demand Drops:** Unexpected loss of large loads.
    -   **Generator Outages:**
        -   Single large generator failure.
        -   Multiple smaller generator failures (cascading or simultaneous).
    -   **High Sensor Noise:** Significant noise or bias in critical sensor readings (e.g., demand forecast, voltage measurements).
    -   **Combined Scenarios:** E.g., a demand peak occurring during a generator outage.

### 3.3. Evaluation Pipeline Stages
1.  **Individual Agent Benchmarking:** Each Safe RL algorithm variant for the ISO agent will be trained and then evaluated across all defined scenarios. Performance will be compared based on the metrics above.
2.  **(If Applicable) Hybrid Agent Combinations:** If the `energy_net` environment supports or requires interaction with other pre-defined intelligent components (beyond simple load/generation models), these combinations will be tested.
3.  **Multi-Agent Validation (ISO + PCS):** Evaluate the performance and safety of the ISO agent when interacting with multiple PCS agents. This will test the system-level stability and coordination.

## 4. Safe RL Algorithms to be Tested (from OmniSafe)

The project proposal explicitly mentions leveraging **OmniSafe**. The following algorithms are candidates for implementation and testing:

1.  **Constrained Policy Optimization (CPO):** A trust-region based algorithm that optimizes the policy while satisfying constraints. Often a strong baseline for safe RL.
2.  **Constrained Update Projection (CUP):** Focuses on worst-case constraint satisfaction by projecting updates into a safe region, particularly handling uncertainty. This aligns well with the project's robustness goal.
3.  **PPO-Lagrangian (PPO-Lag):** Integrates Lagrangian multipliers with Proximal Policy Optimization (PPO) to balance the primary reward objective with constraint satisfaction. Likely a good starting point due to PPO's popularity and stability.
4.  **First Order Constrained Optimization in Policy Space (FOCOPS):** Aims to project the reward-optimized policy into a safe policy space using first-order methods, potentially offering computational advantages.
5.  **Sauté RL:** Augments the state with a "safety budget" and aims for almost-surely safe exploration and exploitation.

**Selection Strategy:**
-   Start with `PPO-Lagrangian` due to its close relationship with PPO (used in `rl-baselines3-zoo`).
-   Implement `CPO` as a well-established baseline.
-   Explore `CUP` for its explicit handling of uncertainty, which is a core project challenge.
-   Evaluate `FOCOPS` and `Sauté RL` based on initial findings and their specific mechanisms for handling constraints and safety.

This plan provides a structured approach to tackling the project. The next immediate step is to install the `omnisafe` dependency. 