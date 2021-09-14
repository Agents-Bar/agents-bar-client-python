# Agents Bar Python Client

This package is a python client for [Agents Bar](https://agents.bar) service.
It isn't supposed to be used in an isolation and you are supposed to have an existing account.
Check the website for more information about the service,
or check the [Agents Bar Docs](https://docs.agents.bar) to learn more how to use this client.

## Quick start

The client allows to communicate with service by wrapping around APIs and coveraging common use patterns.

A common usage is replacing your deep reinforcement learning agent with an entity that uses `step` and `act` APIs for progressing agent and infering action, respectively.
For usage examples as Google Colab check [Doc's quick start](https://docs.agents.bar/getting-started/quick-start.html) link and for scripts check [examples](examples/) directory.

For a minimal (almost) working example check this code snippet:


```python
from agentsbar import Client, RemoteAgent
from agentsbar import environments

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
```

## Support

Agents Bar Client currently supports manipulation of agents, environments and experimnets.
We also provide an abstraction over agent which allows you to use the agent as an object, the same as you are already using it.

The client is intended to be used for easy communication.
Check documentation for all available APIs. In most cases they should be the same as you see in https://agents.bar/docs.

## Installation

### Pip (Recommended)

The latest stable version should always be accessible through `pip` as [agents-bar](https://pypi.org/project/agents-bar). To install locally add `agents-bar` to your dependency file, e.g. requirements.txt, or install it directly using

```
pip install agents-bar
```

### GitHub source

Checkout this package using `git clone git@github.com:agents-bar/agents-bar-client-python`. This will create a new directory `agents-bar-client-python`. Go ahead, enter the directory and install the package via `pip install -e .`.

*Note* we recommend having a separate python environment for standalone projects, e.g. using `python -m venv` command.


## Authentication

To use the client you need to be pass Agents Bar credentials or some proof that you're a user, e.g. `access_token`. There are a few ways how to authenticate your client.

**Note**: Never store your credentials in places easy accessible by others. This includes `git` repositories that have the slightest chance to leave your computer. Definitely nothing that goes to the GitHub/GitLab.

### Environment variables (suggested)

Currently suggested approach for authentication is to set your token or credentials as environment variables.
The client looks first for `AGENTS_BAR_ACCESS_TOKEN` and uses that as its access token.
You can use this approach if you want to login using a different application with securely stored credentials and temporarily set the access token. Otherwise, you can also set your username and password in `AGENTS_BAR_USERNAME` and `AGENTS_BAR_PASSWORD`, respectively.

As an example, in unix, you can set environment variables by using `export` command in shell
```sh
export AGENTS_BAR_ACCESS_TOKEN=<access_token>
... or ...
export AGENTS_BAR_USERNAME=<username>
export AGENTS_BAR_PASSWORD=<password>
```

### Instantiating with credentials

The `RemoteClient` can authenticate using `access_token` or credentials (`username` and `password`) provided when instantiating the agent. 
Only one of these is required and the `access_token` has priority over credentials pair.
Also, note that directly passed variables have priority over the environment variables.


```python
access_token = "<access_token>"
username = "<username>"
password = "<password>"

client = RemoteClient(..., access_token=access_token, username=username, password=password)
```
