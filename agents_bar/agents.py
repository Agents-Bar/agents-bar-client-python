from typing import Dict, List, Optional

from agents_bar.client import Client


def get_agents(client: Client) -> List[Dict]:
    """Gets agents belonging to authenticated user.

    Parameters:
        client (Client): Authenticated client.
    
    Returns:
        List of agents.

    """
    response = client.get('/agents/')
    if not response.ok:
        response.raise_for_status()
    return response.json()

def get_agent(client: Client, agent_name: str) -> Dict:
    """Get indepth information about a specific agent.

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent.
    
    Returns:
        Details of an agent.

    """
    response = client.get(f'/agents/{agent_name}')
    if not response.ok:
        response.raise_for_status()
    return response.json()

def create_agent(client: Client, config: Dict) -> Dict:
    """Creates an agent with specified configuration.

    Parameters:
        client (Client): Authenticated client.
        config (dict): Configuration of an agent.
    
    Returns:
        Details of an agent.

    """
    response = client.post('/agents/', data=config)
    if not response.ok:
        response.raise_for_status()
    return response.json()

def delete_agent(client: Client, agent_name: str) -> bool:
    """Deletes specified agent.

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent to delete.

    Returns:
        Whether agent was delete. True if an agent was delete, False if there was no such agent.

    """
    response = client.delete('/agents/' + agent_name)
    if not response.ok:
        response.raise_for_status()
    return response.status_code == 202


def get_agent_loss(client: Client, agent_name: str) -> Dict:
    """Recent loss metrics.

    Parameters:
        client (Client): Authenticated client.
        agent_name (str): Name of agent.
    
    Returns:
        Loss metrics in a dictionary.

    """
    response = client.get(f'/agents/{agent_name}/loss')
    if not response.ok:
        response.raise_for_status()
    return response.json()

def agent_step(client, agent_name: str, step: Dict) -> None:
    response = client.post(f"/agents/{agent_name}/step", step)
    if not response.ok:
        response.raise_for_status()
    return

def agent_act(client, agent_name: str, obs: Dict, params: Optional[Dict] = None) -> Dict:
    response = client.post(f"/agents/{agent_name}/act", obs, params=params)
    if not response.ok:
        response.raise_for_status()
    return response.json()
