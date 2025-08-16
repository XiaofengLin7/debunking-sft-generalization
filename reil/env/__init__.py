from .sokoban.env import SokobanEnvReil, SokobanEnvReilCardinal, SokobanEnvReilEmoji, SokobanEnvReilEmpty, SokobanEnvReilNumerical, SokobanEnvReilAlphabetical, SokobanEnvReilRandom
from .alfworld.env import ALFWorldTW
from .sokoban.config import SokobanEnvConfig
from .alfworld.config import ALFWorldConfig
from .gp_l.env import GPLEnv, GPLEnvFaceCardsAs10
from .gp_l.config import GPLEnvConfig

REGISTERED_ENVS = {
    'sokoban': SokobanEnvReil,
    'sokoban_cardinal': SokobanEnvReilCardinal,
    'sokoban_emoji': SokobanEnvReilEmoji,
    'sokoban_empty': SokobanEnvReilEmpty,
    'sokoban_numerical': SokobanEnvReilNumerical,
    'sokoban_alphabetical': SokobanEnvReilAlphabetical,
    'sokoban_random': SokobanEnvReilRandom,
    'alfworld': ALFWorldTW,
    'gp_l': GPLEnv,
    'gp_l_face_cards_as_10': GPLEnvFaceCardsAs10,
}

REGISTERED_ENV_CONFIGS = {
    'sokoban': SokobanEnvConfig,
    'sokoban_cardinal': SokobanEnvConfig,
    'sokoban_emoji': SokobanEnvConfig,
    'sokoban_empty': SokobanEnvConfig,
    'sokoban_numerical': SokobanEnvConfig,
    'sokoban_alphabetical': SokobanEnvConfig,
    'sokoban_random': SokobanEnvConfig,
    'alfworld': ALFWorldConfig,
    'gp_l': GPLEnvConfig,
    'gp_l_face_cards_as_10': GPLEnvConfig,
}
    