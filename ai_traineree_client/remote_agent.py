import dataclasses
import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

import requests
from tenacity import retry, stop_after_attempt, after_log
from .types import EncodedAgentState
from .utils import to_list


StateType = List[float]
ActionType = Union[int, List[Union[int, float]]]

SUPPORTED_MODELS = ['dqn', 'ppo', 'ddpg', 'rainbow']  #: Supported models

global_logger = logging.getLogger("Retry")


class RemoteAgent:
    name = "RemoteAgent"
    default_url = "https://agents.bar"
    logger = logging.getLogger("RemoteAgent")

    def __init__(self, agent_name: str, description: str = "", **kwargs):
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
        self.url = self.__parse_url(kwargs)
        # Pop credentials so that they aren't in the app beyond this point
        self.__access_token = self.get_access_token(
            access_token=kwargs.pop("access_token", None), username=kwargs.pop("username", None),
            password=kwargs.pop("password", None)
        )
        self._headers = {"Authorization": f"Bearer {self.__access_token}", "accept": "application/json"}

        self._config: Dict = {}
        self._config.update(**kwargs)

        self._state_size: Optional[int] = kwargs.get('state_size', None)
        self._action_size: Optional[int] = kwargs.get('action_size', None)
        self._agent_model: Optional[str] = kwargs.get('agent_model', None)
        # self.in_features: Tuple[int] = (self.state_size,)
        self.loss: Dict[str, float] = {}

        self.agent_name = agent_name
        self.description = description

    @property
    def state_size(self):
        if self._state_size is None:
            self.sync()
        return self._state_size

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

    @staticmethod
    def __parse_url(kwargs) -> str:
        url = kwargs.get("url", RemoteAgent.default_url)
        if url[:4].lower() != "http":
            url = "https://" + url
        return url + "/api/v1"

    def get_access_token(self, username=None, password=None, access_token=None) -> str:
        """Retrieves access token.

        """
        access_token = access_token if access_token is not None else os.environ.get('AGENTS_BAR_ACCESS_TOKEN')
        if access_token is None:
            access_token = self.__login(username=username, password=password)
        return access_token

    def __login(self, username: Optional[str] = None, password: Optional[str] = None):
        username = username if username is not None else os.environ.get('AGENTS_BAR_USER')
        password = password if password is not None else os.environ.get('AGENTS_BAR_PASS')
        if username is None or password is None:
            raise ValueError("No credentials provided for logging in. Please pass either 'access_token' or "
                             "('username' and 'password'). These credentials should be related to your Agents Bar account.")

        data = dict(username=username, password=password)
        response = requests.post(f"{self.url}/login/access-token", data=data)
        if response.status_code >= 300:
            self.logger.error(response.text)
            raise ValueError(
                f"Received an error while trying to authenticate as username='{username}'. "
                f"Please double check your credentials. Error: {response.text}"
            )
        return response.json()['access_token']

    def create_agent(self, state_size: int, action_size: int, agent_model: str) -> Dict:
        """Creates a new agent in the service.

        Uses provided information on RemoteAgent instantiation to create a new agent.
        Creating a new agent will fail if the owner already has one with the same name.

        *Note* that it can take a few seconds to create a new agent. In such a case,
        any calls to the agent might fail. To make sure that your program doesn't fail
        either use :py:func:`ai_traineree_client.wait_until_agent_exists` or manually sleep for
        a few seconds.

        Parameters:
            state_size (int): Dimensionality of the state space.
            action_size (int): Dimensionality of the action space.
                In case of discrete space, that's a single dimensions with potential values.
                In case of continuous space, that's a number of dimensions in uniform [0, 1] distribution.
            agent_model (str): Name of the model type. Check :py:data:`ai_traineree_client.SUPPORTED_MODELS`
                for accepted values.

        Returns:
            Details of created agent.

        """
        self.__validate_agent_model(agent_model)
        self.agent_model = agent_model
        self._discrete = self.agent_model.lower() in ('dqn', 'rainbow')
        self._config['state_size'] = state_size
        self._config['action_size'] = action_size
        self.logger.debug("Creating an agent (name=%s, model=%s)", self.agent_name, self.agent_model)
        payload = dict(name=self.agent_name, model=self.agent_model, description=self.description, config=self._config)
        response = requests.post(f"{self.url}/agents/", data=json.dumps(payload), headers=self._headers)
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
        response = requests.delete(f"{self.url}/agents/{agent_name}", headers=self._headers)
        if response.status_code >= 300:
            raise RuntimeError(f"Error while deleting the agent '{agent_name}'. Message from server: {response.text}")
        return True

    @property
    def exists(self):
        """Whether the agent service exists and is accessible"""
        response = requests.get(f"{self.url}/agents/{self.agent_name}", headers=self._headers)
        return response.ok
    
    @property
    def is_active(self):
        response = requests.get(f"{self.url}/agents/{self.agent_name}", headers=self._headers)
        if not response.ok:
            response.raise_for_status()
        agent = response.json()
        return agent['is_active']

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

    def info(self):
        """Gets agents meta-data from sever."""
        response = requests.get(f"{self.url}/agents/{self.agent_name}", headers=self._headers)
        return response.json()

    def sync(self) -> None:
        """Synchronizes local information with the one stored in Agents Bar.
        """
        agent = self.info()
        self._config.update(agent['config'])
        self._state_size = agent['config']['state_size']
        self._action_size = agent['config']['action_size']
        self._agent_model = agent['model']

    @retry(stop=stop_after_attempt(3), reraise=True)
    def get_state(self) -> EncodedAgentState:
        """Gets agents state in an encoded snapshot form.

        *Note* that this API has a heavy rate limit.

        Returns:
            Snapshot with config, buffer and network states being encoded.

        """
        response = requests.get(f"{self.url}/snapshot/{self.agent_name}", headers=self._headers)
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
        response = requests.post(f"{self.url}/snapshot/{self.agent_name}", json=j_state, headers=self._headers)
        if not response.ok:
            response.raise_for_status()  # Raises
            return False  # Doesn't reach
        return True

    @retry(stop=stop_after_attempt(3), after=after_log(global_logger, logging.INFO))
    def act(self, state, noise: float = 0) -> ActionType:
        """Asks for action based on provided state.

        Parameters:
            state (List floats): Python list of floats which represent agent's state.
            noise (float): Default 0. Value for epsilon in epsilon-greedy paradigm.

        Returns:
            action (a number or list of numbers): Suggested action to take from this state.
                In case of discrete problems this is a single int value. Otwherise it is
                a list of either floats or ints.

        """
        data = json.dumps(state)
        response = requests.post(f"{self.url}/agents/{self.agent_name}/act?noise={noise}",
                                 data=data, headers=self._headers)
        if not response.ok:
            response.raise_for_status()  # Raises http

        action = response.json()['action']
        if self._discrete:
            return int(action[0])
        return action

    @retry(stop=stop_after_attempt(10), after=after_log(global_logger, logging.INFO))
    def step(self, state: StateType, action: ActionType, reward: float, next_state: StateType, done: bool) -> bool:
        """Providing information from taking a step in environment.

        *Note* that all values have to be python plain values, like ints, floats, lists...
        Unfortunately, numpy, pandas, tensors... aren't currently supported.

        Parameters:
            state (StateType): Current state.
            action (ActionType): Action taken from the current state.
            reward (float): A reward obtained from getting to the next state.
            next_state (StateType): The state that resulted from taking `action` at `state`.
            done (bool): A flag whether the `next_state` is a terminal state.

        """
        step_data = {
            "state": to_list(state), "next_state": to_list(next_state),
            "action": to_list(action), "reward": reward, "done": done,
        }
        data = {"step_data": step_data}

        response = requests.post(f"{self.url}/agents/{self.agent_name}/step",
                                 data=json.dumps(data), headers=self._headers)
        if not response.ok:
            response.raise_for_status()  # Raises http
            return False
        return True
