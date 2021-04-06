from collections import deque
from typing import Tuple

import gym
from ai_traineree_client import RemoteAgent, wait_until_agent_exists


def get_state_action_size(env):
    def __determine_action_size(action_space):
        if "Discrete" in str(type(action_space)):
            return action_space.n
        else:
            return sum(action_space.shape)
    state_shape = env.observation_space.shape
    state_size = state_shape[0] if len(state_shape) == 1 else state_shape
    action_size = __determine_action_size(env.action_space)
    return state_size, action_size


def interact_episode(env: gym.Env, agent: RemoteAgent, eps: float=0, max_iterations: int=10000) -> Tuple[float, int]:
    """Single episode interaction.

    Interacts with the episode until reaching a terminal state.

    Parameters:
        env (gym.Env): The environment with which to interact.
        agent (RemoteAgent): An instance of the AI Traineree agent, likely that's your Remote Agent.
        eps (float): Default 0. Epsilon value in the epsilon-greedy paradigm.
        max_iterations (int): Default 10000. Maximum number of iterations to take before calling quits
            for interacting with the environment.
    
    Returns:
        Tuple of (score, iterations) obtained in given episode. Score is cummulative score,
        i.e. sum of all rewards, and the iterations is the number of iterations taken in the episode.

    """

    score = 0
    state = env.reset().tolist()
    iterations = 0
    done = False

    while(iterations < max_iterations and not done):
        iterations += 1

        action = agent.act(state, eps)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state.tolist()

        score += float(reward)

        agent.step(state, action, reward, next_state, done)

        # n -> n+1  => S(n) <- S(n+1)
        state = next_state

    return score, iterations


reward_goal: float=100.0
max_episodes: int=2000
eps_start: float=1.0
eps_end: float=0.01
eps_decay: float=0.995
window_len: int=100

env_name = 'CartPole-v1'
env = gym.make(env_name)

state_size, action_size = get_state_action_size(env)

agent = RemoteAgent(
    state_size=state_size, action_size=action_size,
    agent_model="dqn", agent_name="DQN_test", description="Description of the agent",
)

# If the agent hasn't been created already, create a new one
if not agent.exists:
    agent.create_agent()
    wait_until_agent_exists(agent=agent)

episode = 0
epsilon = eps_start
mean_scores = []
epsilons = []
scores_window = deque(maxlen=window_len)
all_scores = []

while (episode < max_episodes):
    episode += 1
    print(f"Episode: {episode:03}\t", end="")
    score, iterations = interact_episode(env, agent, epsilon)
    print(f"Score: {score}\tIterations: {iterations}")

    # Keep information for presentation
    scores_window.append(score)
    all_scores.append(score)
    mean_scores.append(sum(scores_window) / len(scores_window))
    epsilons.append(epsilon)

    # Update greedy-epislon value
    epsilon = max(eps_end, eps_decay * epsilon)

    # Stop learning when agent reaches its goal
    if mean_scores[-1] >= reward_goal and len(scores_window) == window_len:
        print(f'Environment solved after {episode} episodes!\tAverage Score: {mean_scores[-1]:.2f}')
        break

