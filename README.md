# CartPole RL Enhancement

This repository contains a solution for an RL code challenge designed for an internship interview process. The project extends the classic CartPole environment from OpenAI Gym by adding a third action ("do nothing") and develops a deep reinforcement learning (RL) solution to handle this modified environment. The purpose of this exercise is to showcase understanding of RL components, environment modification, algorithm development, coding practices, and presentation skills.

## Table of Contents

- [CartPole RL Enhancement](#cartpole-rl-enhancement)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Tasks](#tasks)
  - [Environment Description](#environment-description)
  - [Algorithm Approach](#algorithm-approach)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
    - [Training the Agent](#training-the-agent)
    - [Evaluating the Agent](#evaluating-the-agent)
  - [Results and Analysis](#results-and-analysis)
  - [Extensions and Future Work](#extensions-and-future-work)
  - [Shortcomings](#shortcomings)

## Project Overview

This repository provides a solution to a multi-part RL challenge that involves:

- Modifying the CartPole environment to include a third, "no operation" action.
- Developing and training a deep reinforcement learning agent to learn how to balance the pole with the new action available.
- Implementing curriculum learning and a temperature-scaled softmax exploration strategy.
- Logging various metrics and reward components to TensorBoard.
- Presenting the methodology, environment description, and analysis of the solution.
- (Bonus) Modifying the reward structure to account for energy or effort used by the agent.

## Tasks

The project was broken down into the following tasks:

1. **Task 1:** Modify the CartPole environment to add a third action where the agent does nothing.
2. **Task 2:** Find and implement a deep RL solution using either a custom algorithm or an RL library.
3. **Task 3:** Create a presentation explaining:
   - The modified environment.
   - Steps taken towards the solution.
   - Possible extensions.
   - Potential shortcomings.
4. **Task 4 (Bonus):** Propose and, optionally, implement a new reward function reflecting energy/effort consumption, and analyze its effects.

## Environment Description

The modified environment is based on the classic CartPole scenario. Traditionally, the agent can perform two actions:

- Push the cart to the right.
- Push the cart to the left.

The modification introduces a third action:

- **Do nothing:** The agent takes no action, leaving the system unchanged for that time step.

This change adds complexity to the decision space and allows exploration of how an additional, neutral action influences learning and performance.

The environment also supports a curriculum learning strategy where reward components change over different training phases, and a temperature parameter in the softmax exploration strategy, which decays linearly over time.

## Algorithm Approach

For the deep RL solution, the approach involves:

- Using a deep Q-learning network (DQN) or Dueling DQN to approximate the action-value function.
- Adapting the neural network architecture and training loop to handle three actions instead of two.
- Implementing curriculum learning to adjust reward weights over different training phases.
- Using a temperature-scaled softmax with a linearly decaying temperature for exploration.
- Utilizing a prioritized replay buffer for efficient sampling and training.
- Logging performance metrics and reward components using TensorBoard.
- Scheduling evaluation episodes to monitor agent performance and generalization.
- Modifying the reward structure to include energy usage or effort expended by the agent.

**Note:** While a specific algorithm is used in this repository, the approach is flexible. Other algorithms like PPO, A3C, or custom variants can be substituted depending on preferences and requirements.

## Setup and Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/louisgthier/cartpole-rl-enhancement.git
   cd cartpole-rl-enhancement
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

**Requirements include:**

- Python 3.x
- OpenAI Gymnasium
- PyTorch
- NumPy
- TensorBoard
- Typer
- TorchRL (for Prioritized Replay Buffer)
- Other libraries as specified in requirements.txt

## Usage

The repository uses Typer commands managed through main.py for training and evaluation.

### Training the Agent

To train the RL agent on the modified CartPole environment:

```bash
python main.py train
```

This command will:

- Initialize the modified environment.
- Set up the deep RL model with a prioritized replay buffer, curriculum learning, and temperature scheduling.
- Begin training, periodically saving checkpoints and logging performance metrics.

To resume training from a saved checkpoint:

```bash
python main.py train --resume
```

### Evaluating the Agent

To evaluate a trained model:

```bash
python main.py evaluate experiments/saved_models/checkpoint.pth
```

Replace `experiments/saved_models/checkpoint.pth` with the path to the saved model checkpoint.

## Results and Analysis

After 100k training steps using the equally weighted reward structure (alive, distance to center, pole angle, and energy usage):

- The agent achieved an average reward of 975/1000, indicating near-optimal balancing performance.
- Energy consumption decreased to less than 0.4 per step (full power is 10 per step), demonstrating efficient energy usage.
- The agent reached a 100% success rate during evaluation (1000/1000 steps for every episode), with no failures and potentially infinite episodes.
- Evaluations were conducted every 2k steps, providing a robust monitoring mechanism to assess performance improvements despite off-policy learning, non-deterministic actions, and exploration noise.

Through analysis, it was observed that while prioritized experience replay helped stabilize loss, it did not significantly improve overall performance compared to uniform sampling.

## Extensions and Future Work

Potential extensions to this project include:

- Experimenting with other RL algorithms and comparing their performance.
- Implementing stratified sampling or a distribution-aware buffer, potentially using VAEs.
- Further refining the reward function to incorporate other factors like smoothness of action or energy efficiency.
- Expanding the environment (e.g., adding obstacles, varying physics parameters).
- Hyperparameter tuning for improved performance and stability.
- Increasing the number of parallel environments to 16 for faster experience collection.
- Keeping best results or multiple checkpoints for robust evaluation.
- Scheduling pole angle and distance to center rewards to decrease over time to focus on energy optimization.

## Shortcomings

Possible shortcomings of the current solution:

- The deep RL algorithm may require extensive tuning to reliably balance the pole with three actions.
- The model might overfit to the specific environment configuration, reducing generalizability.
- The "do nothing" action’s usefulness is highly dependent on reward structure and environment dynamics; further adjustments may be needed.
- Computational resources and time required for training deep RL models can be significant.
- The curriculum learning and temperature scheduling parameters may need further tuning for optimal performance.
- Prioritization improved loss stability but did not enhance overall performance significantly.
