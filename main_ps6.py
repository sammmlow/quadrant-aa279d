# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:43:46 2023

Sam Low and Katherine Cao
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm as norm

from source import spacecraft

# Initialize both chief and deputy.
sc1_elements = [6918.14, 0.00001, 97.59760, 0.000000, -109.33800,   0.00827]
sc2_elements = [6918.14, 0.00722, 97.5976, 134.94389, -108.5025, -134.71026] 
sc1 = spacecraft.Spacecraft( elements = sc1_elements )
sc2 = spacecraft.Spacecraft( elements = sc2_elements )