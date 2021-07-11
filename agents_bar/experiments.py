import dataclasses
from typing import Dict, List, Optional

from agents_bar.client import Client
from agents_bar.types import ExperimentCreate
from agents_bar.utils import response_raise_error_if_any

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
    response = client.post(f'{EXP_PREFIX}/', data=dataclasses.asdict(experiment_create))
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
    return response.text
