''' Set default argments
'''
import sys
import os
import ipdb
from glob import glob
from datetime import datetime
import yaml

class Config:
    def __init__(self, 
                 yml_name=None,
                 base_yml_folder=None,):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        # ==== Make a default base yml folder ====]
        if base_yml_folder == None:
            self.base_yml_folder = os.path.join(curr_path, '..', 'configs')
        else:
            self.base_yml_folder = base_yml_folder

        if yml_name == None:
            print('Need to specify a config file')
            ipdb.set_trace()
        else:
            cfg_file = os.path.join(self.base_yml_folder, yml_name)
            if not os.path.exists(cfg_file):
                print('No config file named: %s' % yml_name)
                ipdb.set_trace()
            else:
                self.cfg_file = cfg_file

        # Load the yaml config file
        with open(cfg_file, 'r') as stream:
            cfg = yaml.load(stream)
            # ==== Initialize the variables from the yaml file ====
            self.__dict__.update(cfg)