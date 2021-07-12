from agents_bar.utils import response_raise_error_if_any
from typing import Dict, List, Optional

from agents_bar.client import Client

AGENTS_PREFIX = "/agents"


def get_many(client: Client) -> List[Dict]:
    """Gets agents belonging to authenticated user.

    Parameters:
        client (Client): Authenticated client.
    
    Returns:
        List of agents.

    """
    response = client.get(f'{AGENTS_PREFIX}/')
    response_raise_error_if_any(response)
    return response.json()

def get(client: Client, agent_name: str) -> Dict:
    """Get indepth information about a specific agent.

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent.
    
    Returns:
        Details of an agent.

    """
    response = client.get(f'{AGENTS_PREFIX}/{agent_name}')
    response_raise_error_if_any(response)
    return response.json()

def create(client: Client, config: Dict) -> Dict:
    """Creates an agent with specified configuration.

    Parameters:
        client (Client): Authenticated client.
        config (dict): Configuration of an agent.
    
    Returns:
        Details of an agent.

    """
    response = client.post(f'{AGENTS_PREFIX}/', data=config)
    response_raise_error_if_any(response)
    return response.json()

def delete(client: Client, agent_name: str) -> bool:
    """Deletes specified agent.

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent to delete.

    Returns:
        Whether agent was delete. True if an agent was delete, False if there was no such agent.

    """
    response = client.delete(f'{AGENTS_PREFIX}/' + agent_name)
    response_raise_error_if_any(response)
    return response.status_code == 202


def get_loss(client: Client, agent_name: str) -> Dict:
    """Recent loss metrics.

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent.
    
    Returns:
        Loss metrics in a dictionary.

    """
    response = client.get(f'{AGENTS_PREFIX}/{agent_name}/loss')
    response_raise_error_if_any(response)
    return response.json()

def step(client, agent_name: str, step: Dict) -> None:
    """Steps forward in agents learning mechanism.

    This method is used to provide learning data, like recent observations and rewards,
    and trigger learning mechanism. *Note* that majority of agents perform `step` after
    every environment iteration (`step`).

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent.
        step (Dict): Data required for the agent to use in learning.
            Likely that's `observation`, `next_observation`, `reward`, `action` values and `done` flag.
    
    """
    response = client.post(f"{AGENTS_PREFIX}/{agent_name}/step", data=step)
    response_raise_error_if_any(response)
    return

def act(client, agent_name: str, obs: Dict, params: Optional[Dict] = None) -> Dict:
    """Asks agent about its action on provided observation.

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent.
        obs (Dict): Observation from current environment state.
        params (Optional dict): Anything useful for the agent to learn, e.g. epsilon greedy value.
    
    Returns:
        Dictionary container actions.
    
    """
    response = client.post(f"{AGENTS_PREFIX}/{agent_name}/act", obs, params=params)
    response_raise_error_if_any(response)
    return response.json()
