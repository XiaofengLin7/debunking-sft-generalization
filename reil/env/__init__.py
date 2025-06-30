from .sokoban.env import SokobanEnvReil
from .sokoban.env import SokobanEnvReilCardinal
from .alfworld.env import ALFWorldTW
from .sokoban.config import SokobanEnvConfig
from .alfworld.config import ALFWorldConfig

REGISTERED_ENVS = {
    'sokoban': SokobanEnvReil,
    'sokoban_cardinal': SokobanEnvReilCardinal,
    'alfworld': ALFWorldTW,
}

REGISTERED_ENV_CONFIGS = {
    'sokoban': SokobanEnvConfig,
    'sokoban_cardinal': SokobanEnvConfig,
    'alfworld': ALFWorldConfig,
}
    