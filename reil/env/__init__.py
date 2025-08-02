from .sokoban.env import SokobanEnvReil, SokobanEnvReilCardinal, SokobanEnvReilEmoji, SokobanEnvReilEmpty
from .alfworld.env import ALFWorldTW
from .sokoban.config import SokobanEnvConfig
from .alfworld.config import ALFWorldConfig
from .gp_l.env import GPLEnv
from .gp_l.config import GPLEnvConfig

REGISTERED_ENVS = {
    'sokoban': SokobanEnvReil,
    'sokoban_cardinal': SokobanEnvReilCardinal,
    'sokoban_emoji': SokobanEnvReilEmoji,
    'sokoban_empty': SokobanEnvReilEmpty,
    'alfworld': ALFWorldTW,
    'gp_l': GPLEnv,
}

REGISTERED_ENV_CONFIGS = {
    'sokoban': SokobanEnvConfig,
    'sokoban_cardinal': SokobanEnvConfig,
    'sokoban_emoji': SokobanEnvConfig,
    'sokoban_empty': SokobanEnvConfig,
    'alfworld': ALFWorldConfig,
    'gp_l': GPLEnvConfig,
}
    