#  A Simplistic Market Making Model

In this project, I would like to go over how to build a simplistic Market Making Model to deploy in an Algorithmic Trading context. The code does not contain Reinforcement Learning methods, however it shed some light over how Reinforcement Learning (and Temporal-difference (TD) learning) could help in better train our raw model.

## Sections:


### Introduction on Reinforcement Learning

Reinforcement learning (RL) is an interdisciplinary area of machine learning and optimal control concerned with how an intelligent agent should take actions in a dynamic environment in order to maximize a reward signal. Reinforcement learning is one of the three basic machine learning paradigms, alongside supervised learning and unsupervised learning. The environment is typically stated in the form of a Markov decision process (MDP), as many reinforcement learning algorithms use dynamic programming techniques. The main difference between classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the Markov decision process, and they target large MDPs where exact methods become infeasible, hence the Machine Learning context.


Basic reinforcement learning is modeled as a Markov decision process:

- A set of environment and agent states (the state space), ${\displaystyle {\mathcal {S}}}$:


- ${\displaystyle P_{a}(s,s')=\Pr(S_{t+1}=s'\mid S_{t}=s,A_{t}=a)}$, the transition probability (at time ${\displaystyle t}$) from state ${\displaystyle s}$ to state ${\displaystyle s'}$ under action ${\displaystyle a}$.

- ${\displaystyle R_{a}(s,s')}$, the immediate reward after transition from ${\displaystyle s}$ to ${\displaystyle s'}$ under action ${\displaystyle a}$.


From a Machine Learning standpoint, most of the problems in the fields can be posed as optimisation problems.

We define a loss function $\mathcal{L} : (x, F_w) → \mathbb{R}$
- where $x$ is some data, and $F_w$ is an arbitrary stochastic continuous function parameterised by numeric weights $w$.
- $F$ is often non-linear.
- We minimise the expected loss by choosing appropriate weights: $ argmin _w \mathbb{E}[\mathcal{L}(x, F_w)] $

I also would like to go over Control Theory and specifically Stochastic Optimal Control, which would be essential to fit in our Market Making landscape. 


### Control Theory

Optimal control theory is a branch of control theory that deals with finding a control for a dynamical system over a period of time such that an objective function is optimized. Optimal control is an extension of the calculus of variations, and is a mathematical optimization method for deriving control policies. The method is largely due to the work of Lev Pontryagin and Richard Bellman in the 1950s, after contributions to calculus of variations by Edward J. McShane. Optimal control can be seen as a control strategy in control theory.

Optimal control deals with the problem of finding a control law for a given system such that a certain optimality criterion is achieved. A control problem includes a cost functional that is a function of state and control variables. An optimal control is a set of differential equations describing the paths of the control variables that minimize the cost function. The optimal control can be derived by solving the famous  Hamilton–Jacobi–Bellman equation (a sufficient condition).

In Stochastic Optimal Control problems, a sub-field of Optimal Control, the problems to be solved are in the universe of Stochastic Processes and therefore deal with the existence of uncertainty either in observations or in the noise that drives the evolution of the system. We assume, in a Bayesian probability-driven fashion, that random noise with known probability distribution affects the evolution and observation of the state variables. Stochastic control aims to design the time path of the controlled variables that performs the desired control task with minimum cost, somehow defined, despite the presence of this noise. The context may be either discrete time or continuous time.

An extremely well-studied formulation in stochastic control is that of linear quadratic Gaussian control. Here the model is linear, the objective function is the expected value of a quadratic form, and the disturbances are purely additive.

In a Linear discrete-time stochastic quadratic control problem, we would like to minimize:

${\displaystyle \mathbb {E} _{1}\sum _{t=1}^{S}\left[y_{t}^{\mathsf {T}}Qy_{t}+u_{t}^{\mathsf {T}}Ru_{t}\right]}$

where ${\displaystyle \mathbb {E}}$ is the expected value operator conditional on $y_0$,  S is the time horizon, subject to the state equation:

- ${\displaystyle y_{t}=A_{t}y_{t-1}+B_{t}u_{t},}$

where $y$ is an n × 1 vector of observable state variables, $u$ is a k × 1 vector of control variables, $A_t$ is the time $t$ realization of the stochastic n × n state transition matrix, $B_t$ is the time t realization of the stochastic n × k matrix of control multipliers, and $Q$ (n × n) and $R$ (k × k) are known symmetric positive definite cost matrices.


In a continuous setting, one of the method to attack an Optimal Control problem could be to solve the Hamilton-Jacobi-Bellman Equation.

### The Hamilton-Jacobi-Bellman Equation

The Hamilton-Jacobi-Bellman (HJB) equation is a nonlinear partial differential equation that provides necessary and sufficient conditions for optimality of a control with respect to a loss function. Its solution is the value function of the optimal control problem which, once known, can be used to obtain the optimal control by taking the maximizer (or minimizer) of the Hamiltonian involved in the HJB equation.

We present the HJB equation in a continuous deterministic setting:

${\displaystyle {\frac {\partial V(x,t)}{\partial t}}+\min _{u}\left\{{\frac {\partial V(x,t)}{\partial x}}\cdot F(x,u)+C(x,u)\right\}=0}$

When it comes to extending the problem in a stochastic setting, we would have to consider the below:

${\displaystyle \min _{u}\mathbb {E} \left\{\int _{0}^{T}C(t,X_{t},u_{t})\,dt+D(X_{T})\right\}}$

with:

- ${\displaystyle (X_{t})_{t\in [0,T]}}$ the stochastic process to optimize and
- $ {\displaystyle (u_{t})_{t\in [0,T]}}$ the steering.


Without focussing extensively on the various aspects of an Optimal Control problem, and their Stochastic counterpart, I would like to dive into the Settings for our simple Market Making model, and apply the Reinforcement Learning methodologies in order to find the optimal policies that we aim to identify. 

### Reinforcement Learning and Stochastic Control Problems

Reinforcement learning is technique for solving stochastic control problems. As defined previously, $F_w$ is an arbitrary stochastic continuous function parameterised by numeric weights $w$, often non-linear, which defines a (stochastic) mapping between states of the world, and corresponding
actions to take. The weights w typically specify propensities or probabilities for each action in each state. 

In the context of reinforcement-learning, we refer to the function F as a control policy. The loss function is the negative of the stochastic return obtained by taking the resulting actions. In contrast to traditional stochastic control, we do not have to obtain a closed-form solution for the dynamics of the environment.

#### Learning agents

The entity taking the actions is called the agent.
- The agent repeatedly takes actions $a_t ∈ A$ in discrete time periods $t ∈ \mathbb{N}$.
- When the agent chooses action at at time $t$, it obtains an immediate observable reward $r_{t+1}$.
- The return $G_t$ at time $t$ is some function $f$ of the future rewards $G_t = f (r_{t+1}, r_{t+2}, . . .)$
- Often we use $G_t = \Sigma^{\infty}_{i=t} γ^i r_{i+1}$ where $γ ∈ [0, 1]$ is the time discounting of the agent.


#### Markov Decision Processes
The reward is a function of the action chosen in the previous state $s_t ∈ S$. The $s$ state and action $a$ at time $t$ determines the probability of the subsequent state $s′$ and reward $r$. The probabilities are specified by the function $p(s′, r|s, a)$. This defines a finite Markov Decision-Process and the goal of the agent is to maximise the expected return $\mathbb{E}[G]$; the agent follows a policy which specifies an action to take in each state: $π(s) ∈ A$ and the optimal policy is denoted $π^∗$.

## The Settings for the Market Making Model

In this section, I would like to share some some light over the market making model, that we are going to implement later in our code.

### The market model

We consider three types of agent:
- informed traders,
- uninformed traders, and
- a single market-maker.

Other assumptions that we make are:

- Prices evolve intra-day in discrete time periods $t ∈ \mathbb{N}$.
- A single asset is traded.
- All trades and orders involve a single share of the asset.
- There is no order crossing between traders.
- The arrival of traders at the market follow a Poisson process.


### The Fundamental Price

The true price of the asset p∗t ∈ Z follows a Poisson process. The parameter $λ_p ∈ [0, 1]$ is the probability of a discrete change in the price.



In the simplest form of the model, the market-maker posts a single price $p^m_t$.
The market-maker can adjust its price:

- $ p^m_{t+1} = p_t + \Delta p_t$

where the price changes are discrete and finite (i.e. $ \Delta p_t ∈ {-1,0,1}$).

The reward at time t is the change in the profit:

- for a sell order: $r_t = p^*_t - p^m_t$
- for a buy order: $r_t = p^m_t - p^*_t$

To simplify, we have not considered more complex versions of bid-ask spread and commissions.

#### Informed traders

Informed traders have information about the fundamental price $p^∗_t$ .
- They can submit market orders for immediate execution at the market-maker’s price $p^m_t$
- They submit:
- a buy order if and only if $p^∗_t > p^m_t$
- a sell order if and only if $p^∗_t < p^m_t$
- no order if and only if $p^∗_t = p^m_t$
- They arrive at the market with probability $λ_i$.


#### Uninformed traders
Uninformed traders arrive at the market with probability $2λ_u$.
They submit a buy order for +1 shares with probability $λ_u$, or a sell order for −1 shares with equal probability $λ_u$.


### The Overall Random Process

All Poisson processes are combined resulting in:
- $ 2λ_p + 2λ_u + λ_i = 1 $
There is an event at every discrete time period $t$. Trade occurs a finite period of time $t ∈ {1, 2, . . . , T}$ where T is the duration of a single trading day.
The market maker operates over many days. The initial conditions for every day are the same; they are independent episodes.


#### The market-maker as an agent

We can consider the market-maker as an adaptive agent.
- The environment consists of the observable variables in the market.
- Initially the observable state is the total order-imbalance $IMB_t$.
- The variables are discrete, therefore there is a discrete state space.
- The market-maker chooses actions in discrete time periods $t$.

The set of actions $\mathcal{A}$ available to the agent is $Δp ∈ \mathcal{A} = {−1, 0,+1}$.
It can choose actions conditional on observations in order to maximise expected return $\mathbb{E}[G]$ where $G = \Sigma _t γ^t r_t$.


If you would like to see the full code, please go to the ipynb file!

