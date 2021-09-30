from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from agentsbar.client import Client
from agentsbar.types import LeagueConfig, LeagueCreate
from agentsbar.utils import response_raise_error_if_any

LEAGUE_PREFIX = "/leagues"


def get_many(client: Client) -> List[Dict]:
    """Gets leagues that belong to an authenticated user.

    Parameters:
        client (Client): Authenticated client.
    
    Returns:
        List of leagues.

    """
    response = client.get(f"{LEAGUE_PREFIX}/")
    return response.json()


def get(client: Client, league_name: str) -> Dict:
    """Get indepth information about a specific league.

    Parameters:
        client (Client): Authenticated client.
        league_name (str): Name of league.
    
    Returns:
        Details of an league.

    """
    response = client.get(f'{LEAGUE_PREFIX}/{league_name}')
    response_raise_error_if_any(response)
    return response.json()


def create(client: Client, league_create: LeagueCreate) -> Dict:
    """Creates an league with specified configuration.

    Parameters:
        client (Client): Authenticated client.
        league_create (LeagueCreate): League create instance.
    
    Returns:
        Details of an league.

    """
    response = client.post(f'{LEAGUE_PREFIX}/', data=asdict(league_create))
    response_raise_error_if_any(response)
    return response.json()


def delete(client: Client, league_name: str) -> bool:
    """Deletes specified league.

    Parameters:
        client (Client): Authenticated client.
        league_name (str): Name of the league.

    Returns:
        Whether league was delete. True if an league was delete, False otherwise.

    """
    response = client.delete(f'{LEAGUE_PREFIX}/{league_name}')
    response_raise_error_if_any(response)
    return response.status_code == 202


def reset(client: Client, league_name: str) -> str:
    """Resets the league to starting position.

    Doesn't affect Agent nor Environment. Only resets values related to the League.

    Parameters:
        client (Client): Authenticated client.
        league_name (str): Name of the league.
    
    Returns:
        Confirmation on reset league.

    """
    response = client.post(f"{LEAGUE_PREFIX}/{league_name}/reset")
    response_raise_error_if_any(response)
    return response.json()


def start(client: Client, league_name: str, config: Optional[Dict] = None) -> str:
    """Starts league, i.e. creates experiments with provided agents and environments.

    Parameters:
        client (Client): Authenticated client.
        league_name (str): Name of the league.
    
    Returns:
        Information about started league.
    
    """
    config = config or {}
    response = client.post(f"{LEAGUE_PREFIX}/{league_name}/start", data=config)
    response_raise_error_if_any(response)
    return "Started successfully" if response.ok else "Failed to start"


def metrics(
    client: Client, league_name: str, metric_names: Optional[List[str]] = None, limit: int = 1,
) -> Dict[str, List[Tuple[int, float]]]:
    """Gets metrics obtained while running an league.

    Parameters:
        client (Client): Authenticated client.
        league_name (str): Name of the league.
        metric_names (Optional list of strings): List of metrics you are intrested in seeing.
            If None then it'll return all available. Defaults to None.
        limit (int): Number of last samples to return. Defaults to only the most recent metrics.
    
    Returns:
        Dictionary with keys being metric names and values in a list consisting of an index and value (tuple).
        For example:
            {
                "episode/score": [(10, -10), (9, -7), (8, 3)],
                "loss/actor": [(10, 1.5), (9, 4.1), (8, 0.99)]
            }
    
    """
    response = client.post(f"{LEAGUE_PREFIX}/{league_name}/metrics", data=metric_names, params=dict(limit=limit))
    response_raise_error_if_any(response)
    return response.json()
