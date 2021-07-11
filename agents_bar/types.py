from dataclasses import Field, dataclass
from typing import Any, List, Dict, Optional, Union

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
class ExperimentCreate:
    name: str
    agent_name: str
    environment_name: str
    config: Dict[str, Any]
    description: Optional[str] = None
    is_active: Optional[bool] = True
