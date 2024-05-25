from chessml import script
import omegaconf

@script
def train(args, config):
    print(omegaconf.OmegaConf.to_yaml(config))
