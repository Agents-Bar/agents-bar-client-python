from typing import Any, Dict, List

from agents_bar.client import Client
from agents_bar.utils import response_raise_error_if_any

ENV_PREFIX = "/environments"


def get_many(client: Client) -> List[Dict]:
    """Gets environments belonging to authenticated user.

    Parameters:
        client (Client): Authenticated client.
    
    Returns:
        List of environments.

    """
    response = client.get(f"{ENV_PREFIX}/")
    return response.json()

def get(client: Client, env_name: str) -> Dict:
    """Get indepth information about a specific environment.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of environment.
    
    Returns:
        Details of an environment.

    """
    response = client.get(f'{ENV_PREFIX}/{env_name}')
    response_raise_error_if_any(response)
    return response.json()

def create(client: Client, config: Dict) -> Dict:
    """Creates an environment with specified configuration.

    Parameters:
        client (Client): Authenticated client.
        config (dict): Configuration of an environment.
    
    Returns:
        Details of an environment.

    """
    response = client.post(f'{ENV_PREFIX}/', data=config)
    response_raise_error_if_any(response)
    return response.json()

def delete(client: Client, env_name: str) -> bool:
    """Deletes specified environment.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of the environment.

    Returns:
        Whether environment was delete. True if an environment was delete, False if there was no such environment.

    """
    response = client.delete(f'{ENV_PREFIX}/{env_name}')
    response_raise_error_if_any(response)
    return response.status_code == 202

def reset(client: Client, env_name: str) -> List[float]:
    """Resets the environment to starting position.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of the environment.
    
    Returns:
        Environment state in the starting position.

    """
    response = client.post(f"{ENV_PREFIX}/{env_name}/reset")
    response_raise_error_if_any(response)
    return response.json()


def step(client: Client, env_name: str, step) -> Dict[str, Any]:
    """Steps the environment based on provided data.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of the environment.
        step (EnvStep): Step information for the environment. Consists of action details and whether to commit.
    
    Returns:
        Environment state after taking provided actions. Consists of "observation", "reward", "done" and "info".

    """
    response = client.post(f"{ENV_PREFIX}/{env_name}/step", data=step)
    response_raise_error_if_any(response)
    return response.json()


def commit(client: Client, env_name: str) -> Dict[str, Any]:
    """Commits last provided data. Must be proceeded by environment `step`.

    Useful when environment requires many agents or when agent is allowed to make mistakes.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of the environment
    
    Returns:
        Data about the state the environment has transtioned into.
        This should be the same as when using `step` with `commit=True`.
    
    """
    response = client.post(f"{ENV_PREFIX}/{env_name}/commit")
    response_raise_error_if_any(response)
    return response.json()

def info(client: Client, env_name: str) -> Dict[str, Any]:
    response = client.get(f"{ENV_PREFIX}/{env_name}/info")
    response_raise_error_if_any(response)
    return response.json()
