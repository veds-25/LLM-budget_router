## Project Description - DevSoc 

In this project, we will build a RL based routing system that learns how to allocate a fixed inference budget across a stream of incoming requests. Given the characteristics of each query and the remaining budget, the agent must decide which model to use in order to maximize overall response quality throughout the entire episode.

You will design and train a Reinforcement Learning agent that routes incoming queries to different LLMs while operating under a global token budget of 10k tokens *[1 token is 1 word]*. The objective is to maximize overall system performance by balancing two competing goals:

- **Average response quality**
- **Number of queries successfully answered**

You are free to design the reward function, state representation, routing strategy, and evaluation methodology. The project encourages exploration of different RL formulations and trade offs between quality and coverage under limited resources.

## Dataset

https://huggingface.co/datasets/JiaqiXue/mmr-routing-20k

## Learning Goals

- Formulate real-world problems as RL environments.
- Design effective states, actions, and rewards.
- Understand budget-aware decision making and quality–cost trade-offs.
- Evaluate the impact of reward design on learned policies.
- Benchmark RL agents against heuristic baselines.
- Gain hands-on experience training and analyzing RL systems.
