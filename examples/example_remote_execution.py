# Example of how to create an agent and an environment.
# Once we create these, we then communicate them (locally) so that agent interacts with the environment.
from agents_bar.types import ExperimentCreate
from agents_bar import Client, RemoteAgent
from agents_bar import environments, experiments

# Define client to communicate with https://agents.bar. Make sure it's authenticated.
client = Client()

# Create an environment. Simple one is "CartPole-v1" from OpenAI gym repo.
env_name = "CartPole"
environments.create(client, config={"name": env_name, "config": {"gym_name": "CartPole-v1"}})

# Create an agent. Since environment is discrete we use DQN.
agent_name = "CartPoleAgent"
agent = RemoteAgent(client, agent_name=agent_name)
agent.create_agent(obs_size=4, action_size=2, agent_model="DQN")

# Create an Experiment which allows for Agent <-> Environment communication
exp_name = "CartPoleExperiment"
experiment_create = ExperimentCreate(
    name=exp_name, agent_name=agent_name, environment_name=env_name, config={}, description="Testing experimnt on CartPole"
)
experiments.create(client, experiment_create)

# One command allows to start the communication.
# After this the whole learning process is done in Agents Bar.
experiments.start(client, exp_name)
