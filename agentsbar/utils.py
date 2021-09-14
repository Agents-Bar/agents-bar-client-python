import time
from typing import List

import requests
from requests.models import HTTPError


SUPPORTED_ENTITIES = ('agent', 'environment', 'experiment')


def wait_until_active(client, entity: str, name: str, max_seconds: int = 20, verbose: bool = True) -> bool:
    """
    Waits until the agent is created but no longer than `max_seconds`.

    Parameters:
        client (Client): Authenticated client.
        entity (str): Currently only 'agent', 'environment' and 'experiment' are supported.
        name (str): Name of the entity, e.g. name of the Agent you're trying to check.
        max_seconds (int): Maximum seconds allowed to wait. Default: 20 (seconds).
        verbose (bool): Whether to print logs to standard output. Default: True.

    Returns:
        Boolean value, whether agent exists, i.e. was successfully created.
    """
    assert entity in SUPPORTED_ENTITIES, f"Only '{SUPPORTED_ENTITIES}' are supported"
    start_time = time.time()
    elapsed_time = 0

    while elapsed_time < max_seconds:
        response = client.get(f'/{entity}s/{name}')
        if response.ok and response.json()['is_active']:
            break

        if verbose and elapsed_time:
            print(f"Waited {elapsed_time:0.2f} seconds. Waiting some more...")
        time.sleep(0.5)
        elapsed_time = time.time() - start_time

    return True


def wait_until_exists(client, entity: str, name: str, max_seconds: int = 20, verbose: bool = True) -> bool:
    """
    Waits until the agent is created but no longer than `max_seconds`.

    Parameters:
        client (Client): Authenticated client.
        entity (str): Currently only 'agent', 'environment' and 'experiment' are supported.
        name (str): Name of the entity, e.g. name of the Agent you're trying to check.
        max_seconds (int): Maximum seconds allowed to wait. Default: 20 (seconds).
        verbose (bool): Whether to print logs to standard output. Default: True.

    Returns:
        Boolean value, whether agent exists, i.e. was successfully created.
    """
    assert entity in SUPPORTED_ENTITIES, f"Only '{SUPPORTED_ENTITIES}' are supported"
    start_time = time.time()
    elapsed_time = 0

    while elapsed_time < max_seconds:
        response = client.get(f'/{entity}s/{name}')
        if response.ok:
            break

        if verbose and elapsed_time:
            print(f"Waited {elapsed_time:0.2f} seconds. Waiting some more...")
        time.sleep(0.5)
        elapsed_time = time.time() - start_time

    return True


def wait_until_agent_is_active(agent, max_seconds: int = 20, verbose: bool = True) -> bool:
    """
    Waits until the agent is is_active but no longer than `max_seconds`.

    Parameters:
        agent (RemoteAgent): Remote agent instance.
        max_seconds (int): Maximum seconds allowed to wait.
        verbose:

    Returns:
        Boolean value, whether agent is_active, i.e. exists and is ready to respond.
    
    """
    return wait_until_active(agent._client, 'agent', agent.agent_name, max_seconds=max_seconds, verbose=verbose)


def wait_until_agent_exists(agent, max_seconds: int = 20, verbose: bool = True) -> bool:
    """
    Waits until the agent is created but no longer than `max_seconds`.

    Parameters:
        agent (RemoteAgent): Remote agent instance.
        max_seconds (int): Maximum seconds allowed to wait.
        verbose:

    Returns:
        Boolean value, whether agent exists, i.e. was successfully created.
    """
    return wait_until_exists(agent._client, 'agent', agent.agent_name, max_seconds=max_seconds, verbose=verbose)

def to_list(x: object) -> List:
    """Convert to a list.

    Parameters:
        x (object): Something that would make sense converting to a list.

    Returns:
        Tries to create a list from provided object.

    Examples:
        >>> to_list(1)
        [1]
        >>> to_list([1,2])
        [1, 2]
        >>> to_list( (1.2, 3., 0.) )
        [1.2, 3., 0.]

    """
    if isinstance(x, list):
        return x
    if isinstance(x, (int, float)):
        return [x]
    # Just hoping...
    return list(x)


def response_raise_error_if_any(response: requests.Response) -> None:
    """
    Checks if there is any error while make a request.
    If status 400+ then raises HTTPError with provided reason.
    """
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        msg = response.text
        try:
            msg = response.json().get('detail')
        except:
            pass
        raise HTTPError({"error": str(e), "reason": msg}) from None
