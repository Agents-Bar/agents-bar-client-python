from dataclasses import dataclass
from typing import List, Union


@dataclass
class EncodedAgentState:
    model: str
    obs_size: int
    action_size: int
    encoded_config: str
    encoded_network: str
    encoded_buffer: str

ObsType = List[float]
ActionType = Union[int, List[Union[int, float]]]
