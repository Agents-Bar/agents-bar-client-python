from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from agentsbar.client import Client
from agentsbar.types import ExperimentCreate
from agentsbar.utils import response_raise_error_if_any

EXP_PREFIX = "/experiments"


def get_many(client: Client) -> List[Dict]:
    """Gets experiments that belong to an authenticated user.

    Parameters:
        client (Client): Authenticated client.
    
    Returns:
        List of experiments.

    """
    response = client.get(f"{EXP_PREFIX}/")
    return response.json()


def get(client: Client, exp_name: str) -> Dict:
    """Get indepth information about a specific experiment.

    Parameters:
        client (Client): Authenticated client.
        exp_name (str): Name of experiment.
    
    Returns:
        Details of an experiment.

    """
    response = client.get(f'{EXP_PREFIX}/{exp_name}')
    response_raise_error_if_any(response)
    return response.json()


def create(client: Client, experiment_create: ExperimentCreate) -> Dict:
    """Creates an experiment with specified configuration.

    Parameters:
        client (Client): Authenticated client.
        config (dict): Configuration of an experiment.
    
    Returns:
        Details of an experiment.

    """
    response = client.post(f'{EXP_PREFIX}/', data=asdict(experiment_create))
    response_raise_error_if_any(response)
    return response.json()


def delete(client: Client, exp_name: str) -> bool:
    """Deletes specified experiment.

    Parameters:
        client (Client): Authenticated client.
        exp_name (str): Name of the experiment.

    Returns:
        Whether experiment was delete. True if an experiment was delete, False if there was no such experiment.

    """
    response = client.delete(f'{EXP_PREFIX}/{exp_name}')
    response_raise_error_if_any(response)
    return response.status_code == 202


def reset(client: Client, exp_name: str) -> str:
    """Resets the experiment to starting position.

    Doesn't affect Agent nor Environment. Only resets values related to the Experiment,
    like keeping score of last N episodes or managing Epislon value.

    Parameters:
        client (Client): Authenticated client.
        exp_name (str): Name of the experiment.
    
    Returns:
        Confirmation on reset experiment.

    """
    response = client.post(f"{EXP_PREFIX}/{exp_name}/reset")
    response_raise_error_if_any(response)
    return response.json()


def start(client: Client, exp_name: str, config: Optional[Dict] = None) -> str:
    """Starts experiment, i.e. communication between selected Agent and Env entities.

    Parameters:
        client (Client): Authenticated client.
        exp_name (str): Name of the experiment.
    
    Returns:
        Information about started experiment.
    
    """
    config = config or {}
    response = client.post(f"{EXP_PREFIX}/{exp_name}/start", data=config)
    response_raise_error_if_any(response)
    return "Started successfully" if response.ok else "Failed to start"


def metrics(
    client: Client, exp_name: str, metric_names: Optional[List[str]] = None, limit: int = 1,
    ) -> Dict[str, List[Tuple[int, float]]]:
    """Gets metrics obtained while running an experiment.

    Parameters:
        client (Client): Authenticated client.
        exp_name (str): Name of the experiment.
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
    response = client.post(f"{EXP_PREFIX}/{exp_name}/metrics", data=metric_names, params=dict(limit=limit))
    response_raise_error_if_any(response)
    return response.json()
