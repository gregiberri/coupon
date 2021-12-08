#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : Albert Gregus
@Email   : g.albert95@gmail.com
"""

import argparse
import logging
from config import ConfigNamespace
from ml.solvers.hpo_solver import HPOSolver
from ml.solvers.decision_tree_solver import DecisionTreeSolver
from ml.solvers.pytorch_tabular_solver import PytorchTabularSolver
from utils.device import DEVICE
from utils.init_random_seeds import set_random_seed

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--id_tag', type=str, default='base', help='Id of the training in addition of the config name')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'hpo'],
                    help='The mode of the running.')
parser.add_argument('-c', '--config', type=str, default='random_forest', help='Config file name')

args = parser.parse_args()

if __name__ == '__main__':
    config = ConfigNamespace(args.config)
    set_random_seed(config.env.random_seed)
    logging.info(f'Using {DEVICE} for running.')

    # run a simple training/val/test or an HPO
    if args.mode == 'hpo':
        solver = HPOSolver(config, args)
    elif config.env.solver == 'pytorch_tabular':
        solver = PytorchTabularSolver(config, args)
    elif config.env.solver == 'decision_tree':
        solver = DecisionTreeSolver(config, args)
    else:
        raise ValueError(f'Wrong setting.')

    solver.run()
