from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class ALFWorldConfig:
    train_eval: str = "eval_out_of_distribution"
    render_mode: str = "complete"
    config_path: str = "./reil/env/alfworld/base_config.yaml"
    prefix: Optional[str] = None