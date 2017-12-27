# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import importlib

last_level_dir_path = os.path.abspath(os.curdir+"/../")
sys.path.insert(0, last_level_dir_path)
importlib.reload(sys)

import trainer
from darkflow.cli import cliHandler

cliHandler(sys.argv)

