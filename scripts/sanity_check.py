from chessml import script, config
import omegaconf

@script
def train(args):
    print(omegaconf.OmegaConf.to_yaml(config))
