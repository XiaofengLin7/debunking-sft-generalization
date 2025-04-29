from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class ALFWorldConfig:
    train_eval: str = "eval_out_of_distribution"
    render_mode: str = "complete"