from lib.models.depthdetr import build_depthdetr


def build_model(cfg):
    return build_depthdetr(cfg)
