# AI Traineree Client

This package is a client for the [Agents Bar](https://agents.bar). It isn't supposed to be used
in an isolation. Check the website for more information about the service, or check the 
[Agents Bar Docs](https://docs.agents.bar) for more specific information on how to use the client.

## Quick start

The client uses `step` and `act` methods to operate with the environment. Using `act` you ask
the agent about the action to take, and using `step` you update the agent with you learned.
For examples check [Agent's Bar doc](https://docs.agents.bar/getting-started/quick-start.html) for Google Colab link or go to [examples](examples/).

For a minimal (almost) working example check this code snippet:


```python
env = gym.make('CartPole-v1')  # OpenAI Gym
agent = RemoteAgent(obs_size=4, action_size=2, agent_model="DQN", agent_name="DQN_Test")
obs = env.reset().tolist()

for iteration in range(10):
    action = agent.act(obs, eps)

    next_obs, reward, done, _ = env.step(action)
    next_obs = next_obs.tolist()

    agent.step(obs, action, reward, next_obs, done)
    obs = next_obs

```

## Installation

### Pip (Recommended)

The latest stable version should always be accessible through `pip` at [ai-traineree-client](https://pypi.org/project/ai-traineree-client). To install locally add `ai-traineree-client` to your dependency file, e.g. requirements.txt, or install it directly using

```
pip install ai-traineree-client
```

### GitHub source

Checkout this package using `git clone git@github.com:laszukdawid/ai-traineree-client.git`. This will create a new directory `ai-traineree-client`. Go ahead, enter the directory and install the package via `pip install -e .`.

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
