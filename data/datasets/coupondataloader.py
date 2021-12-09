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

        self.df = self.load_data()
        self.handle_nans()
        self.embed_categoricals()

        # make numpy arrays to train
        self.pd_outputs = self.df.Y
        self.outputs = self.pd_outputs.to_numpy()
        self.pd_inputs = self.df.drop(columns=['Y', 'Unnamed: 0'])
        self.inputs = self.pd_inputs.to_numpy()

    @property
    def data(self):
        return self.inputs, self.outputs

    @property
    def df_data(self):
        return self.pd_inputs, self.pd_outputs

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

        # drop index to avoid doubleing
        df.drop(columns=['Unnamed: 0'], inplace=True)

        train, val = train_test_split(df, test_size=0.33)
        train.to_csv(os.path.join(self.config.data_dir, 'train.csv'))
        val.to_csv(os.path.join(self.config.data_dir, 'val.csv'))

    def handle_nans(self):
        """
        Either delete or fill nans (with `nan`)
        :return:
        """
        # drop direction_opp as it is the same as direction_same
        self.df.drop(columns='direction_opp', inplace=True)

        if self.config.remove_nans:
            # drop missing values
            self.df.drop(columns='car', inplace=True)
            self.df.dropna(inplace=True)
        else:
            self.df.fillna('nan', inplace=True)

    def embed_categoricals(self):
        """
        Embed categorical columns to ordinal or one_hot embedding.
        """
        if self.config.ordinal_embed: self.df = make_ordinal(self.df, self.config.ordinal_embedding_columns)
        if self.config.one_hot_embed: self.df = make_one_hot(self.df, self.config.one_hot_embedding_columns)

