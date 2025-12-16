# InforMARL Copilot Instructions

This repository implements **InforMARL**, a Graph Neural Network (GNN) framework for Multi-Agent Reinforcement Learning (MARL) with limited local observability.

## Project Architecture

The codebase is structured into three main components:
1.  **Environment (`multiagent/`)**: Based on Multi-Agent Particle Environment (MPE).
    -   `MPE_env.py`: Entry point for creating environments (`MPEEnv`, `GraphMPEEnv`).
    -   `environment.py`: Base classes `MultiAgentBaseEnv` and `MultiAgentGraphEnv`.
    -   `core.py`: Physics engine and entity definitions (`World`, `Agent`, `Landmark`).
    -   `custom_scenarios/`: Defines specific tasks (rewards, observations, world setup).
2.  **Algorithms (`onpolicy/`)**: Implementation of on-policy RL algorithms (MAPPO, RMAPPO).
    -   `scripts/train_mpe.py`: Main training entry point.
    -   `runner/`: Handles the training loop (`Shared/Separated` runners).
    -   `algorithms/`: PPO implementation details.
    -   `config.py`: Centralized configuration arguments.
3.  **Scripts**:
    -   `train_scripts/`: Shell scripts to launch training experiments.
    -   `render_scripts/`: Scripts for visualizing trained policies.

## Critical Workflows

### Training
-   **Entry Point**: `onpolicy/scripts/train_mpe.py`.
-   **Execution**: Use shell scripts in `train_scripts/` which set necessary arguments.
    ```bash
    # Example: Train 5 agents in formation
    cd train_scripts_ablation
    sh train_formation_5agts.sh
    ```
-   **Configuration**: Arguments are defined in `onpolicy/config.py` and overridden via command line in shell scripts.

### Environment Development
-   **Scenarios**: To create a new task, add a script in `multiagent/custom_scenarios/`.
    -   Must implement `Scenario` class with `make_world`, `reset_world`, `reward`, and `observation`.
    -   For Graph environments, also implement `graph_observation` and `update_graph`.
-   **Graph Integration**: `GraphMPEEnv` (in `multiagent/MPE_env.py`) uses `MultiAgentGraphEnv` to handle graph-structured observations.

### Rendering
-   Use scripts in `render_scripts/`.
-   Ensure `use_render` is set to `True` (or passed as arg) and `render_episodes` is specified.

## Conventions & Patterns

-   **Global Variables**: The project uses `onpolicy.global_var` (imported as `glv`) to share state like curriculum learning ratios across modules.
-   **Callbacks**: The environment relies heavily on callbacks (`reset_callback`, `reward_callback`, etc.) defined in the scenario file and passed to the environment constructor.
-   **Argument Passing**: A monolithic `args` namespace object is passed through most functions and classes to carry configuration.
-   **Graph Observations**:
    -   `graph_observation_callback` returns `node_obs` and `adj` (adjacency matrix).
    -   `update_graph` modifies the graph structure dynamically based on agent positions/states.

## Dependencies
-   **GNN**: `torch_geometric` is used for graph operations.
-   **RL**: Custom PPO implementation in `onpolicy/`.
-   **Logging**: `wandb` is used for experiment tracking.

## Common Issues
-   **Path Handling**: Scripts often assume the working directory is the project root or specific subdirectories. Check `sys.path.append` in scripts.
-   **Environment Versions**: `MPEEnv` vs `GraphMPEEnv` - ensure the correct one is instantiated based on `env_name` argument.
