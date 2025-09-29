from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class GPLEnvConfig:
    target: int = 24
    num_cards: int = 4
    treat_face_cards_as_10: bool = True
    ood: bool = False