from dataclasses import dataclass

@dataclass
class EncodedAgentState:
    model: str
    state_size: int
    action_size: int
    encoded_config: str
    encoded_network: str
    encoded_buffer: str
