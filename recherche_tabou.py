# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:15:47 2024

@author: damoi
"""

import numpy as np

#%% Recherche Tabou


def recherche_tabou(sqrt_matrix, r):
    a = sqrt_matrix.shape
    #best_S=np.zeros(a)
    best_mask=np.ones(a)
    best_r = r
    
    mask = np.ones(a) 
    for i in range(a[0]):
        for j in range(a[1]):
            mask[i][j] = -1 * mask[i][j]
            
            U, S, Vh = np.linalg.svd(sqrt_matrix*mask)
            rank = np.size(S)
            
            print(f"mask \n {mask}")
            print(f"rank {rank}")
            
            if np.allclose((((U@S)@Vh)), sqrt_matrix, atol=1e-8) and rank < r:
                best_mask = mask
                best_r = rank
            
            mask[i][j] = -1 * mask[i][j]
            
            
    return best_mask, best_r
            
            
#%%

matrix = np.array([[9, 4],[16,9]])

def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        n, m =map(int,fin.readline().strip().split())

        while True:
            tmp=fin.readline().strip().split()
            matrix.append(list(map(float,tmp)))
    return np.array(matrix)

matrix = read_matrix("exempleslide_matrice.txt")
sqrt_matrix=np.zeros(matrix.shape)
sqrt_matrix=np.sqrt(matrix)

rank = np.linalg.matrix_rank(sqrt_matrix)

best_mask, best_r = recherche_tabou(sqrt_matrix, rank)
print(f"best mask \n {best_mask}")
print(f"best rank {best_r}")