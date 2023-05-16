# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:30:25 2023

By: Sam Low and Katherine Cao
"""

from math import sin, cos, pi, sqrt
from source import spacecraft

def compute_roe(sc1, sc2):
    da = (sc2.a - sc1.a) / sc1.a
    dL = (sc2.M + sc2.w) - (sc1.M + sc1.w) + (sc2.R - sc1.R) * cos(sc1.i)
    dex = sc2.e * cos(sc2.w) - sc1.e * cos(sc1.w)
    dey = sc2.e * sin(sc2.w) - sc1.e * sin(sc1.w)
    dix = sc2.i - sc1.i
    diy = (sc2.R - sc1.R) * sin(sc1.i)
    ROE = [da, dL, dex, dey, dix, diy]
    print('QSN ROEs of SC2: ', ROE)
    return ROE

if __name__ == '__main__':
    sc1_elements = [6918.14, 0.00001, 97.59760, 0.000000, -109.33800,   0.00827]
    sc2_elements = [6918.14, 0.00722, 97.5976, 134.94389, -108.5025, -134.71026]
    sc3_elements = [6918.14, 0.00723, 98.42577, 44.944, -109.338, -44.84428]
    sc1 = spacecraft.Spacecraft( elements = sc1_elements )
    sc2 = spacecraft.Spacecraft( elements = sc2_elements )
    sc3 = spacecraft.Spacecraft( elements = sc3_elements )
    compute_roe(sc1, sc2)
    compute_roe(sc1, sc3)