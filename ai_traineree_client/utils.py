import time


def wait_until_agent_exists(agent, max_seconds: int=20, verbose: bool=True) -> bool:
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


def to_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, (int, float)):
        return [x]
    else:
        # Just hoping...
        return list(x)
