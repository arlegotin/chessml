from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Optional
import logging
from yaml import safe_load
from pathlib import Path
from omegaconf import OmegaConf

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


class Script:
    def __init__(self, path_to_config: Optional[Path] = None):
        self.parser = ArgumentParser()
        self.path_to_config = path_to_config

    def add_argument(self, *args, **kwargs) -> None:
        self.parser.add_argument(*args, **kwargs)

    def run(self, fn: Callable[[Namespace, Dict[str, Optional[str]]], None]) -> None:
        if self.path_to_config:
            config = OmegaConf.load(self.path_to_config)
        else:
            config = OmegaConf.create()

        logging.basicConfig(level=int(config.logs.level))

        parsed_args = self.parser.parse_args()

        logger.info("executing script")

        for key, value in vars(parsed_args).items():
            logger.info(f"{key}: {value}")

        fn(parsed_args, config)

    __call__ = run


script = Script(path_to_config=Path("./config.yaml"))
