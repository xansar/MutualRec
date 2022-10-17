#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   weight_init.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/15 13:11   zxx      1.0         None
"""

# import lib
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    # elif isinstance(m, nn.Embedding):
    #     nn.init.kaiming_uniform_(m.weight.data)