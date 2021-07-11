import time
from typing import List

import requests
from requests.models import HTTPError


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
    start_time = time.time()
    elapsed_time = 0

    while not agent.is_active:
        if verbose and elapsed_time:
            print(f"Waited {elapsed_time:0.2f} seconds. Waiting some more...")
        time.sleep(0.5)
        elapsed_time = time.time() - start_time
        if elapsed_time > max_seconds:
            return False

    return True


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
    start_time = time.time()
    elapsed_time = 0

    while not agent.exists:
        if verbose and elapsed_time:
            print(f"Waited {elapsed_time:0.2f} seconds. Waiting some more...")
        time.sleep(0.5)
        elapsed_time = time.time() - start_time
        if elapsed_time > max_seconds:
            return False

    return True


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
        raise HTTPError({"error": str(e), "reason": response.json()['detail']}) from None
