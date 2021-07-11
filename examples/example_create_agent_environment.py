# Example of how to create an agent and an environment.
# Once we create these, we then communicate them (locally) so that agent interacts with the environment.
from agents_bar import Client, RemoteAgent
from agents_bar import environments

# Define client to communicate with https://agents.bar. Make sure it's authenticated.
client = Client()

# Create an environment. Simple one is "CartPole-v1" from OpenAI gym repo.
env_name = "CartPole"
environments.create(client, config={"name": env_name, "config": {"gym_name": "CartPole-v1"}})

# Create an agent. Since environment is discrete we use DQN.
agent = RemoteAgent(client, agent_name="CartPoleAgent")
agent.create_agent(obs_size=4, action_size=2, agent_model="DQN")

# Initiat learning loop. Observe env's state, pass to agent, make a decision (action), execute on env. Repeat.
obs = environments.reset(client, env_name)
for iteration in range(10):
    action = agent.act(obs)

    out = environments.step(client, env_name, step={"actions": [action], "commit": True})
    next_obs, reward, done = out.get("observation"), out.get("reward"), done.get("done")

    agent.step(obs, action, reward, next_obs, done)
    obs = next_obs