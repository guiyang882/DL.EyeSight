# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configparser

def process_config(conf_file):
    """process configure file to generate
    CommonParams, DataSetParams, NetParams

    Returns:
        CommonParams, DataSetParams, NetParams, SolverParams
    """

    common_params = dict()
    dataset_params = dict()
    net_params = dict()
    solver_params = dict()

    # configure parser
    config = configparser.ConfigParser()
    config.read(conf_file)

    # sections and options
    for section in config.sections():
        if section == "Common":
            for option in config.options(section):
                common_params[option] = config.get(section, option)
        if section == "DataSet":
            for option in config.options(section):
                dataset_params[option] = config.get(section, option)
        if section == "Net":
            for option in config.options(section):
                net_params[option] = config.get(section, option)
        if section == "Solver":
            for option in config.options(section):
                solver_params[option] = config.get(section, option)

    return common_params, dataset_params, net_params, solver_params

