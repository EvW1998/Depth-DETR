import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4):
    """
    cfg:
        type: &dataset_type 'KITTI'
        root_dir: 'data/KITTIDataset'
        train_split: 'train'
        test_split: 'val'
        batch_size: 16
        use_3d_center: True
        class_merging: False
        use_dontcare: False
        bbox2d_type: 'anno'  # 'proj' or 'anno'
        meanshape: False  # use predefined anchor or not
        writelist: ['Car']
        clip_2d: False

        aug_pd: True
        aug_crop: True

        random_flip: 0.5
        random_crop: 0.5
        scale: 0.05
        shift: 0.05

        depth_scale: 'normal'
    """

    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, test_loader
