# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'
import numpy as np
# 恢复兼容性（临时方案，不推荐用于长期项目）
np.bool = bool
np.int = int
np.float = float
np.object = object
np.str = str
np.complex = complex
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='ARS', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

    config_dict = {
        'gpu_id': 0,

    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


