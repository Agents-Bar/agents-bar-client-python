# Example of how to create an agent and an environment.
# Once we create these, we then communicate them (locally) so that agent interacts with the environment.
from agents_bar.utils import wait_until_active, wait_until_agent_is_active
from agents_bar import Client, RemoteAgent, environments
from agents_bar.types import DataSpace, EnvironmentCreate

# Define client to communicate with https://agents.bar. Make sure it's authenticated.
client = Client()

# Create an environment. Simple one is "CartPole-v1" from OpenAI gym repo.
env_name = "CartPole"
env_create = EnvironmentCreate(name=env_name, image='agents-bar/env-gym', config={"gym_name": "CartPole-v1"})
environments.create(client, env_create)
wait_until_active(client, 'environment', env_name)

# Create an agent. Since environment is discrete we use DQN.
agent = RemoteAgent(client, agent_name="CartPoleAgent")
agent_obs_space = DataSpace(dtype='float', shape=(4,))
agent_action_space = DataSpace(dtype='int', shape=(1,), low=0, high=2)
agent.create_agent(obs_space=agent_obs_space, action_space=agent_action_space, agent_model="DQN")
wait_until_agent_is_active(agent)

# Initiat learning loop. Observe env's state, pass to agent, make a decision (action), execute on env. Repeat.
obs = environments.reset(client, env_name)
for iteration in range(10):
    action = agent.act(obs)

    out = environments.step(client, env_name, step={"actions": [action], "commit": True})
    next_obs, reward, done = out.get("observation"), out.get("reward"), out.get("done")

    agent.step(obs, action, reward, next_obs, done)
    obs = next_obs
