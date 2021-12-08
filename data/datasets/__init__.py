from torch.utils.data import DataLoader

from data.datasets.coupondataloader import CouponDataloader


def get_dataloader(data_config, mode):
    # get the iterator object
    if data_config.name == 'coupon_data':
        dataset = CouponDataloader(data_config.params, mode)
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    return dataset
