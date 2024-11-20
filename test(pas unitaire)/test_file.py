# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:36:51 2024

@author: benja
"""

import numpy as np

import math

def opentestfile (path):
    
    with open(path,'r') as fin:
        
        nbr_line,nbr_column=map(int,fin.readline().split())
        input_matrix=[]
        
        for i in range(nbr_line):
            
            line=list(map(int,fin.readline().split()))
            input_matrix.append(line)
           
        input_matrix=np.array(input_matrix)
        
    return input_matrix

a=opentestfile('exempleslide_matrice.txt')


