# -*- coding: utf-8 -*-
__version__ = "0.1.0"

# Was this imported from setup.py?
try:
    __PAPARAZZI_SETUP__
except NameError:
    __PAPARAZZI_SETUP__ = False

# Import all modules
if not __PAPARAZZI_SETUP__:

    # Import the main interface
    from .doppler import Doppler
    from . import utils
