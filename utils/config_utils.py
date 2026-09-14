import math
import os

from loguru import logger
from omegaconf import OmegaConf


def _repo_root() -> str:
    """Absolute path of the gear_sonic / sonic_wbt directory.

    Used in yaml paths via ``${repo_root:}/data/...`` so file references are
    independent of the cwd from which the entry script is launched. Resolved
    by walking up from this file (``utils/config_utils.py``).
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def register_rl_resolvers():

    try:
        OmegaConf.register_new_resolver("eval", eval)
        OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
        OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
        OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
        OmegaConf.register_new_resolver("sum", lambda x: sum(x))
        OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
        OmegaConf.register_new_resolver("int", lambda x: int(x))
        OmegaConf.register_new_resolver("len", lambda x: len(x))
        OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
        OmegaConf.register_new_resolver("repo_root", _repo_root)
    except Exception as e:
        logger.warning(f"Warning: Some resolvers already registered: {e}")
