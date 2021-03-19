from ofa import imagenet_codebase
from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock
from torch.nn.modules.module import Module

def build_block(block_config):

    return MobileInvertedResidualBlock.build_from_config(block_config)
