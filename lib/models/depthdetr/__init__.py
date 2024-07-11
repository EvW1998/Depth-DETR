from .depthdetr import build


def build_depthdetr(cfg):
    return build(cfg)
