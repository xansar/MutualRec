#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config_parser.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/13 15:42   zxx      1.0         None
"""

from configparser import ConfigParser

class MyConfigParser(ConfigParser):
    def __init__(self,defaults=None):
        super(MyConfigParser, self).__init__()
    def optionxform(self, optionstr):
        return optionstr
