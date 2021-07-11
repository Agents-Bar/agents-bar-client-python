import dataclasses
import logging
from typing import Any, Dict, Optional, Union

from tenacity import after_log, retry, stop_after_attempt, wait_fixed

from agents_bar.client import Client

from .types import ActionType, EncodedAgentState, ObsType
from .utils import response_raise_error_if_any, to_list

SUPPORTED_MODELS = ['dqn', 'ppo', 'ddpg', 'rainbow']  #: Supported models

global_logger = logging.getLogger("Global")


class RemoteAgent:
    name = "RemoteAgent"
    default_url = "https://agents.bar"
    logger = logging.getLogger("RemoteAgent")

    def __init__(self, client: Client, agent_name: str,  **kwargs):
        """
        An instance of the agent in the Agents Bar.

        Parameters:
            description (str): Optional. Description for the model, if creating a new one.

        Keyword arguments:
            access_token (str): Default None. Access token to use for authentication. If none provided
                then one is obtained by logging to the service using credentials.
            username (str): Default None. Overrides username from the env variables.
            password (str): Default None. Overrides password from the env variables.

        """
        self._client: Client = client

        self._config: Dict = {}
        self._config.update(**kwargs)

        self._discrete: Optional[bool] = None
        self._obs_size: Optional[int] = kwargs.get('obs_size', None)
        self._action_size: Optional[int] = kwargs.get('action_size', None)
        self._agent_model: Optional[str] = kwargs.get('agent_model', None)
        self.loss: Dict[str, float] = {}

        self.agent_name = agent_name
        self._description: Optional[str] = None

    @property
    def obs_size(self):
        if self._obs_size is None:
            self.sync()
        return self._obs_size

    @property
    def action_size(self):
        if self._action_size is None:
            self.sync()
        return self._action_size

    @property
    def agent_model(self):
        if self._agent_model is None:
            self.sync()
        return self._agent_model

    def create_agent(self, obs_size: int, action_size: int, agent_model: str, active: bool = True, description: Optional[str] = None) -> Dict:
        """Creates a new agent in the service.

        Uses provided information on RemoteAgent instantiation to create a new agent.
        Creating a new agent will fail if the owner already has one with the same name.

        *Note* that it can take a few seconds to create a new agent. In such a case,
        any calls to the agent might fail. To make sure that your program doesn't fail
        either use :py:func:`agents_bar.wait_until_agent_exists` or manually sleep for
        a few seconds.

        Parameters:
            obs_size (int): Dimensionality of the observation space.
            action_size (int): Dimensionality of the action space.
                In case of discrete space, that's a single dimensions with potential values.
                In case of continuous space, that's a number of dimensions in uniform [0, 1] distribution.
            agent_model (str): Name of the model type. Check :py:data:`agents_bar.SUPPORTED_MODELS`
                for accepted values.
            active (bool): Whether to activate the agent.

        Returns:
            Details of created agent.

        """
        self.__validate_agent_model(agent_model)
        self._agent_model = agent_model
        self._discrete = None
        self._config['obs_size'] = obs_size
        self._config['action_size'] = action_size
        self._description = description

        self.logger.debug("Creating an agent (name=%s, model=%s)", self.agent_name, self.agent_model)
        payload = dict(
            name=self.agent_name,
            model=self.agent_model,
            description=self._description,
            config=self._config,
            is_active=active,
        )
        response = self._client.post('/agents', data=payload)
        # response = requests.post(f"{self.url}/agents/", data=json.dumps(payload), headers=self._headers)
        if response.status_code >= 300:
            raise RuntimeError("Unable to create a new agent.\n%s" % response.json())
        return response.json()

    def remove(self, *, agent_name: str, quite: bool = True) -> bool:
        """Deletes the agent.

        **Note** that this action is irreversible. All information about agent will be lost.

        Parameters:
            agent_name (str): You are required to pass the name of the agent as
                              a proof that you're an adult and you know what you're doing.
            quite (bool): Silently ignores if provided agent_name doesn't match actual name.

        Returns:
            Boolean whether delete was successful.

        """
        if agent_name is None or self.agent_name != agent_name:
            if quite:
                self.logger.warning("You're request for deletion is being ignored. You're welcome.")
                return False
            raise ValueError("You wanted to delete an agent. Are you sure? If so, we need *again* its name.")

        self.logger.warning("Agent '%s' is being exterminated", agent_name)
        response = self._client.delete(f"/agents/{agent_name}")
        if response.status_code >= 300:
            raise RuntimeError(f"Error while deleting the agent '{agent_name}'. Message from server: {response.text}")
        return True

    @property
    def exists(self):
        """Whether the agent service exists and is accessible"""
        response = self._client.get(f"/agents/{self.agent_name}")
        return response.ok
    
    @property
    def is_active(self):
        response = self._client.get(f"/agents/{self.agent_name}")
        if not response.ok:
            response.raise_for_status()
        agent = response.json()
        return agent['is_active']

    @property
    def discrete(self):
        if self._discrete is None:
            self._discrete = self.agent_model.lower() in ("dqn", 'rainbow')
        return self._discrete

    @staticmethod
    def __validate_agent_model(model):
        if model.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' isn't currently supported. Please select one from {SUPPORTED_MODELS}")

    @property
    def hparams(self) -> Dict[str, Union[str, float, int]]:
        """Agents hyperparameters

        Returns:
            Dictionary of agent's hyperparameters.
            Values are either numbers or strings, even if they could be different.

        """
        def make_str_or_number(val):
            return str(val) if not isinstance(val, (int, float)) else val

        return {k: make_str_or_number(v) for (k, v) in self._config.items()}

    def info(self) -> Dict[str, Any]:
        """Gets agents meta-data from sever."""
        response = self._client.get(f"/agents/{self.agent_name}")
        info = response.json()
        self._config = info.get('config', self._config)
        return info

    def sync(self) -> None:
        """Synchronizes local information with the one stored in Agents Bar.
        """
        agent = self.info()
        self._agent_model = agent['model']
        self._config.update(agent['config'])
        # TODO: Remove str key once migrated to obs_space
        obs_key = "obs_size" if "obs_size" in self._config else "obs_space"
        action_key = "action_size" if "action_size" in self._config else "action_space"
        self._obs_size = self._config.get(obs_key)
        self._action_size = self._config.get(action_key)

    @retry(stop=stop_after_attempt(3), reraise=True)
    def get_state(self) -> EncodedAgentState:
        """Gets agents state in an encoded snapshot form.

        *Note* that this API has a heavy rate limit.

        Returns:
            Snapshot with config, buffer and network states being encoded.

        """
        response = self._client.get(f"/snapshots/{self.agent_name}")
        if not response.ok:
            response.raise_for_status()
        state = response.json()
        return EncodedAgentState(**state)
    
    def upload_state(self, state: EncodedAgentState) -> bool:
        """Updates remote agent with provided state.

        Parameters:
            state: Agent's state with encoded values for buffer, config and network states.

        Returns:
            Bool confirmation whether update was successful.

        """
        j_state = dataclasses.asdict(state)
        response = self._client.post(f"/snapshots/{self.agent_name}", data=j_state)
        if not response.ok:
            response.raise_for_status()  # Raises
            return False  # Doesn't reach
        return True

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(0.01), after=after_log(global_logger, logging.INFO))
    def act(self, obs, noise: float = 0) -> ActionType:
        """Asks for action based on provided observation.

        Parameters:
            obs (List floats): Python list of floats which represent agent's observation.
            noise (float): Default 0. Value for epsilon in epsilon-greedy paradigm.

        Returns:
            action (a number or list of numbers): Suggested action to take from this observation.
                In case of discrete problems this is a single int value. Otwherise it is
                a list of either floats or ints.

        """
        response = self._client.post(f"/agents/{self.agent_name}/act", params={"noise": noise}, data=obs)
        if not response.ok:
            response.raise_for_status()  # Raises http

        action = response.json()['action']
        if self.discrete:
            return int(action[0])
        return action

    @retry(stop=stop_after_attempt(5),  wait=wait_fixed(0.01), after=after_log(global_logger, logging.INFO), reraise=True)
    def step(self, obs: ObsType, action: ActionType, reward: float, next_obs: ObsType, done: bool) -> bool:
        """Providing information from taking a step in environment.

        *Note* that all values have to be python plain values, like ints, floats, lists...
        Unfortunately, numpy, pandas, tensors... aren't currently supported.

        Parameters:
            obs (ObsType): Current observation.
            action (ActionType): Action taken from the current observation.
            reward (float): A reward obtained from getting to the next observation.
            next_obs (ObsType): The observation that resulted from taking `action` at `obs`.
            done (bool): A flag whether the `next_obs` is a terminal state.

        """
        step_data = {
            "obs": to_list(obs), "next_obs": to_list(next_obs),
            "action": to_list(action), "reward": reward, "done": done,
        }
        data = {"step_data": step_data}

        response = self._client.post(f"/agents/{self.agent_name}/step", data=data)
        response_raise_error_if_any(response)
        return True
