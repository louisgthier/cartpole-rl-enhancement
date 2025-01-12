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

## Algorithm Approach
For the deep RL solution, the approach involves:
- Using a deep Q-learning network (DQN) to approximate the action-value function.
- Adapting the neural network architecture and training loop to handle three actions instead of two.
- Using a prioritized replay buffer for efficient sampling and training.
- Evaluating the agent’s performance based on balancing the pole and, optionally, accounting for energy/effort using a modified reward function.

**Note:** While a specific algorithm is used in this repository, the approach is flexible. Other algorithms like PPO, A3C, or custom variants can be substituted depending on preferences and requirements.

## Setup and Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/RL-CartPole-Challenge.git
cd RL-CartPole-Challenge
```

2. **Create and activate a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
   - Other libraries as specified in `requirements.txt`

## Usage
The repository uses Typer commands managed through `main.py` for training and evaluation.

### Training the Agent
To train the RL agent on the modified CartPole environment:
```bash
python main.py train
```
This command will:
- Initialize the modified environment.
- Set up the deep RL model with a prioritized replay buffer.
- Begin training, periodically saving checkpoints and logging performance metrics.

To resume training from a saved checkpoint:
```bash
python main.py train --resume
```

### Evaluating the Agent
To evaluate a trained model:
```bash
python main.py evaluate path/to/saved_model
```
Replace `path/to/saved_model` with the actual path to the model file you want to evaluate.

## Results and Analysis
After training:
- The agent's performance metrics (e.g., episode lengths, rewards) are logged using TensorBoard.
- Analysis of how the third action affects learning and performance is documented.
- (If bonus was attempted) Comparison of training results before and after modifying the reward to account for energy usage.

## Extensions and Future Work
Potential extensions to this project include:
- Experimenting with other RL algorithms and comparing their performance.
- Further refining the reward function to incorporate other factors like smoothness of action or energy efficiency.
- Expanding the environment (e.g., adding obstacles, varying physics parameters).
- Hyperparameter tuning for improved performance and stability.

## Shortcomings
Possible shortcomings of the current solution:
- The deep RL algorithm may require extensive tuning to reliably balance the pole with three actions.
- The model might overfit to the specific environment configuration, reducing generalizability.
- The "do nothing" action’s usefulness is highly dependent on reward structure and environment dynamics; further adjustments may be needed.
- Computational resources and time required for training deep RL models can be significant.
