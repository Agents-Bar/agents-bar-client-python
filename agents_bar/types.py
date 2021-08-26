from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

ObsType = List[float]
ActionType = Union[int, List[Union[int, float]]]


@dataclass
class EncodedAgentState:
    model: str
    obs_size: int
    action_size: int
    encoded_config: str
    encoded_network: str
    encoded_buffer: str


@dataclass
class AgentCreate:
    name: str
    model: str
    image: str
    config: Dict[str, Any]
    description: Optional[str] = None
    is_active: Optional[bool] = True


@dataclass
class EnvironmentCreate:
    name: str
    image: str
    config: Dict[str, Any]
    description: Optional[str] = None
    is_active: Optional[bool] = True


@dataclass
class ExperimentCreate:
    name: str
    agent_names: List[str]
    environment_names: List[str]
    config: Dict[str, Any]
    description: Optional[str] = None
    is_active: Optional[bool] = True


@dataclass
class Space:
    dtype: Optional[str] = None
    shape: Optional[Tuple[int]] = None
    low: Optional[Union[int, float, List[Any]]] = None
    high: Optional[Union[int, float, List[Any]]] = None
