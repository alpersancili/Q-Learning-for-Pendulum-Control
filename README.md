# Q-Learning-for-Pendulum-Control
This project implements a Q-Learning algorithm to solve the Pendulum-v1 environment from OpenAI's Gymnasium library. The objective is to train an agent to balance a pendulum in an upright position using a discretized state and action space.

PROJECT OVERVIEW

Reinforcement learning is used to tackle the continuous control problem of the Pendulum-v1 environment. The algorithm employs a discretization approach to transform the continuous state and action spaces into discrete segments, enabling the application of Q-Learning—a tabular RL method.
The project includes:

Training a Q-Learning agent to maximize rewards in the Pendulum environment.
Visualization of performance over episodes with rolling averages.
Saving and reloading the Q-table for testing and further training.

ENVIRONMENT DETAILS

Environment: Pendulum-v1
State Space: Continuous (3D vector representing angle, velocity, and angular velocity).
Action Space: Continuous (torque applied to the pendulum).
Objective: Apply torque to keep the pendulum upright and minimize energy expenditure.

KEY FEATURES

State and Action Discretization:
The continuous state and action spaces are divided into discrete bins to make the problem solvable with Q-Learning.

Epsilon-Greedy Exploration:
Balances exploration and exploitation using a decaying epsilon strategy.

Reward Tracking:
Rewards are recorded per episode.
Rolling averages of rewards over the last 100 episodes are plotted for performance evaluation.

Q-Table Saving and Loading:
The Q-table is saved when the agent achieves a new best reward. This enables testing the trained policy or resuming training later.

Visualization:
Generates a reward graph showing the learning progress.
Supports rendering the environment during testing.

FILES IN THE REPOSITORY

pendulum_q.py: Main Python script containing the implementation.
pendulum.pkl: Saved Q-table (generated after training).
pendulum.png: Reward graph showing training performance.
