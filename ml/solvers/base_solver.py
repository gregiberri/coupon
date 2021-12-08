import gc
import logging
import sys
import time
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.datasets import get_dataloader
from ml.models import get_model
from ml.modules.losses import get_loss
from ml.optimizers import get_optimizer, get_lr_policy, get_lr_policy_parameter
from utils.device import DEVICE, put_minibatch_to_device
from utils.iohandler import IOHandler


class Solver(object):

    def __init__(self, config, args):
        """
        Solver parent function to control the experiments.
        It contains everything for an experiment to run.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        self.args = args
        self.phase = args.mode
        self.config = config
        self.save_preds = self.config.env.save_preds

        # initialize the required elements for the ml problem
        self.init_dataloaders()
        self.init_model()
        self.iohandler = IOHandler(args, self)

    def init_model(self):
        """
        Initialize the model according to the config and put it on the gpu if available,
        (weights can be overwritten during checkpoint load).
        """
        logging.info("Initializing the model.")
        self.model = get_model(self.config.model)

    def init_dataloaders(self):
        """
        Dataloader initialization(s) for train, val dataset according to the config.
        """
        logging.info("Initializing dataloaders.")
        if self.phase == 'train':
            self.train_loader = get_dataloader(self.config.data, 'train')
            self.val_loader = get_dataloader(self.config.data, 'val')
        elif self.phase in ['val', 'test']:
            self.val_loader = get_dataloader(self.config.data, self.phase)
        else:
            raise ValueError(f'Wrong mode argument: {self.phase}. It should be `train`, `val` or `test`.')

    def run(self):
        """
        Run the experiment.
        :return: the best goal metrics (as stated in config.metrics.goal_metric).
        """
        logging.info("Starting experiment.")
        if self.phase == 'train':
            self.train()
        elif self.phase in ['val', 'test']:
            self.eval()
        else:
            raise ValueError(f'Wrong phase: {self.phase}. It should be `train`, `val` or `test`.')

        return self.accuracy

    def train(self):
        """
        Training the tree based networks.
        Save the model if it has better performance than the previous ones.
        """
        self.model.fit(*self.train_loader.full_data)
        self.eval()

        # self.iohandler.save_best_checkpoint()
        # self.iohandler.writer.close()

    def eval(self):
        """
        Evaluate the model.
        """
        preds = self.model.predict(self.val_loader.full_data[0])
        gt = self.val_loader.full_data[1]

        self.accuracy = accuracy_score(gt, preds)
        print(self.accuracy)

        if self.save_preds: self.iohandler.save_preds()