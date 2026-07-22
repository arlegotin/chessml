from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Optional
import logging
from pathlib import Path
from omegaconf import OmegaConf

__version__ = "0.1.1"

logger = logging.getLogger(__name__)

config = OmegaConf.merge(
    OmegaConf.load(Path("./config.yaml")),
    OmegaConf.load(Path("./config.local.yaml")),
)

class Script:
    def __init__(self):
        self.parser = ArgumentParser()

    def add_argument(self, *args, **kwargs) -> None:
        self.parser.add_argument(*args, **kwargs)

    def run(self, fn: Callable[[Namespace, Dict[str, Optional[str]]], None]) -> None:
        logging.basicConfig(level=int(config.logs.level))

        parsed_args = self.parser.parse_args()

        logger.info("executing script")

        for key, value in vars(parsed_args).items():
            logger.info(f"{key}: {value}")

        seed = getattr(parsed_args, "seed", None)
        if seed is None:
            seed = getattr(parsed_args, "shuffle_seed", None)

        if seed is not None:
            from lightning import seed_everything

            seed_everything(seed, workers=True)

        fn(parsed_args)

    __call__ = run


script = Script()
