#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run.py.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/16 16:02   zxx      1.0         None
"""

# import lib
import torch
import numpy as np

import argparse
import random

from utils import *

def get_config(config_pth):
	config = MyConfigParser()
	config.read('./config/' + config_pth, encoding='utf-8')
	return config._sections

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)

def parse_args():
	########
	# Parses the MutualRec arguments.
	#######
	parser = argparse.ArgumentParser(description="Run NJM.")

	parser.add_argument('--config_pth', type=str, default='MutualRec_debug.ini' ,
						help='Choose config')
	return parser.parse_args()

def run(config_pth):
	config = get_config(config_pth)
	seed = eval(config['TRAIN']['random_seed'])
	setup_seed(seed)

	dataset = Epinions()
	model = MutualRec(config)
	model.apply(weight_init)
	lr = eval(config['OPTIM']['learning_rate'])
	weight_decay = eval(config['OPTIM']['weight_decay'])
	optimizer = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
	loss_func = BPRLoss(eval(config['LOSS']['balance_factor']))
	metric = MutualRecMetirc(ks=eval(config['METRIC']['ks']))

	trainer = Trainer(
		model=model,
		loss_func=loss_func,
		optimizer=optimizer,
		metric=metric,
		dataset=dataset,
		config=config
	)
	trainer.train()

if __name__ == '__main__':
	args = parse_args()
	run(args.config_pth)