from dataclasses import dataclass


@dataclass
class Outcome:
    p: float
    new_state: int
    reward: float = 0.0
    is_terminal: bool = False
