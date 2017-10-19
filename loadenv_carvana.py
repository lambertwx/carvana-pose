# -*- coding: utf-8 -*-
"""
Loads the environment for my Carvana stuff.

$Id$
"""

import importlib
import os
from pathlib import Path
from pprint import pprint

# This assumes that the GPU server, which will be shared by different users, each with their own
# repositories, won't have c:/safe.  Maybe someday we'll have to figure out a different approach.
safepath = Path("c:/safe")

if safepath.is_dir():
    base = "c:/safe"
else:
    base = os.getenv("HOMEPATH")

# Files that define classes


# Everything else - these may have type declarations that use the above classes
runfile(base + "/svn/vidmisc/trunk/vision2017/common/apply_hysteresis_threshold.py", ".")

from common.imagetile import extracttiles, layouttiles
from common.imgutils import *
from common.videofig import redraw_fn, videofig

from carvana_prep import *
