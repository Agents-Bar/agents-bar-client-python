from typing import Any, Dict, List

from agents_bar.client import Client


def environment_get_all(client: Client) -> List[Dict]:
    """Gets environments belonging to authenticated user.

    Parameters:
        client (Client): Authenticated client.
    
    Returns:
        List of environments.

    """
    response = client.get('/env/')
    if not response.ok:
        response.raise_for_status()
    return response.json()

def environment_get(client: Client, env_name: str) -> Dict:
    """Get indepth information about a specific environment.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of environment.
    
    Returns:
        Details of an environment.

    """
    response = client.get(f'/env/{env_name}')
    if not response.ok:
        response.raise_for_status()
    return response.json()

def environment_create(client: Client, config: Dict) -> Dict:
    """Creates an environment with specified configuration.

    Parameters:
        client (Client): Authenticated client.
        config (dict): Configuration of an environment.
    
    Returns:
        Details of an environment.

    """
    response = client.post('/env/', data=config)
    if not response.ok:
        response.raise_for_status()
    return response.json()

def environment_delete(client: Client, env_name: str) -> bool:
    """Deletes specified environment.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of the environment.

    Returns:
        Whether environment was delete. True if an environment was delete, False if there was no such environment.

    """
    response = client.delete('/env/' + env_name)
    if not response.ok:
        response.raise_for_status()
    return response.status_code == 202

def environment_reset(client: Client, env_name: str) -> List[float]:
    """Resets the environment to starting position.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of the environment.
    
    Returns:
        Environment state in the starting position.

    """
    response = client.post(f"/env/{env_name}/reset")
    if not response.ok:
        response.raise_for_status()
    return response.json()

def environment_step(client: Client, env_name: str, step) -> Dict[str, Any]:
    """Steps the environment based on provided data.

    Parameters:
        client (Client): Authenticated client.
        env_name (str): Name of the environment.
        step (EnvStep): Step information for the environment. Consists of action details and whether to commit.
    
    Returns:
        Environment state after taking provided actions. Consists of "observation", "reward", "done" and "info".

    """
    response = client.post(f"/env/{env_name}/step", data=step)
    if not response.ok:
        response.raise_for_status()
    return response.json()
