from .sokoban.env import SokobanEnvReil
from .alfworld.env import ALFWorldTW
from .sokoban.config import SokobanEnvConfig
from .alfworld.config import ALFWorldConfig

REGISTERED_ENVS = {
    'sokoban': SokobanEnvReil,
    'alfworld': ALFWorldTW,
}

REGISTERED_ENV_CONFIGS = {
    'sokoban': SokobanEnvConfig,
    'alfworld': ALFWorldConfig,
}
    