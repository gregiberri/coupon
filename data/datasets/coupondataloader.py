import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data.utils.handle_categorical import make_ordinal, make_one_hot


class CouponDataloader:
    """
    Dataloader to load the traffic signs.
    """

    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.data = self.load_data()
        self.embed_categoricals()

        # make numpy arrays to train
        self.outputs = self.data.Y.to_numpy()
        inputs_pd = self.data.drop(columns=['Y', 'Unnamed: 0'])
        self.inputs = inputs_pd.to_numpy()

    @property
    def full_data(self):
        return self.inputs, self.outputs

    def load_data(self):
        """
        Load data from the corresponding csv file.

        :return: data pandas dataframe
        """
        logging.info(f'Loading {self.split} data')

        if self.split == 'val':
            filepath = os.path.join(self.config.data_dir, 'val.csv')
        elif self.split == 'train':
            filepath = os.path.join(self.config.data_dir, 'train.csv')
        else:
            raise ValueError(f'Wrong split: {self.split}.')

        if not os.path.exists(filepath):
            self.train_val_split()

        return pd.read_csv(filepath)

    def train_val_split(self):
        """
        Split coupons_data.csv to train-val set and save to train.csv and val.csv
        """
        coupons_filepath = os.path.join(self.config.data_dir, 'coupon_data.csv')
        if not os.path.exists(coupons_filepath):
            raise ValueError(f'The coupon_data file does not exists at {coupons_filepath}')

        df = pd.read_csv(coupons_filepath)
        # drop missing values
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df.drop(columns='car', inplace=True)
        df.dropna(inplace=True)
        df.drop(columns='direction_opp', inplace=True)

        train, val = train_test_split(df, test_size=0.33)
        train.to_csv(os.path.join(self.config.data_dir, 'train.csv'))
        val.to_csv(os.path.join(self.config.data_dir, 'val.csv'))

    def embed_categoricals(self):
        """
        Embed categorical columns to ordinal or one_hot embedding.
        """
        self.data = make_ordinal(self.data, self.config.ordinal_embedding_columns)
        self.data = make_one_hot(self.data, self.config.one_hot_embedding_columns)

