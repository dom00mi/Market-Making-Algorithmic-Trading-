#  A Simplistic Market Making Model

In this project, I would like to go over how to build a simplistic Market Making Model to deploy in an Algorithmic Trading context. The code does not contain Reinforcement Learning methods, however it shed some light over how Reinforcement Learning (and Temporal-difference (TD) learning) could help in better train our raw model.

## Sections:


### Introduction on Reinforcement Learning

Reinforcement learning (RL) is an interdisciplinary area of machine learning and optimal control concerned with how an intelligent agent should take actions in a dynamic environment in order to maximize a reward signal. Reinforcement learning is one of the three basic machine learning paradigms, alongside supervised learning and unsupervised learning. The environment is typically stated in the form of a Markov decision process (MDP), as many reinforcement learning algorithms use dynamic programming techniques. The main difference between classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the Markov decision process, and they target large MDPs where exact methods become infeasible, hence the Machine Learning context.


Basic reinforcement learning is modeled as a Markov decision process:

- A set of environment and agent states (the state space), ${\displaystyle {\mathcal {S}}}$:


- ${\displaystyle P_{a}(s,s')=\Pr(S_{t+1}=s'\mid S_{t}=s,A_{t}=a)}$, the transition probability (at time ${\displaystyle t}$) from state ${\displaystyle s}$ to state ${\displaystyle s'}$ under action ${\displaystyle a}$.

- ${\displaystyle R_{a}(s,s')}$, the immediate reward after transition from ${\displaystyle s}$ to ${\displaystyle s'}$ under action ${\displaystyle a}$.

  
![newplot (13)](https://github.com/user-attachments/assets/15b5ff52-9ceb-4cac-b373-f9f053147d67)


If you would like to see the full code, please go to the ipynb file!

