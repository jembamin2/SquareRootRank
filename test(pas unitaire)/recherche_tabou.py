# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:15:47 2024

@author: damoi
"""

import numpy as np
import hashlib

#%%

def rankof(array, tol=10**-10):
    count=0
    for item in array:
        if item>tol:
            count+=1
    return count

def hash_matrix(matrix):
    # Convertir la matrice en une chaîne et calculer un hash
    matrix_str = str(matrix)
    return hashlib.md5(matrix_str.encode()).hexdigest()

def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        n, m =map(int,fin.readline().strip().split())
        
        tmp=fin.readline().strip().split()
        while tmp:
            matrix.append(list(map(float,tmp)))
            tmp=fin.readline().strip().split()
    return np.array(matrix)

def voisinage(masks, matrix, mask, r):
    best_mask = mask
    best_r = r
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            mask[i][j] = -1 * mask[i][j]
            mask_hash = hash_matrix(mask)

            # Vérifier si la matrice existe déjà dans le dictionnaire via son hash
            if mask_hash not in masks:
                masks[mask_hash] = mask  # Stocker la matrice seulement si elle est nouvelle
            
                U,S,Vh = np.linalg.svd(matrix*mask, full_matrices= False)
                rank = rankof(S)
                S = np.diag(S)
                
                
                if np.allclose((((U@S)@Vh))**2, matrix**2, atol=1e-10):
                    if rank <= r:
                        print(f" new rank {rank}")
                        print(f" new mask {mask}")
                        best_mask = mask.copy()
                        best_r = rank
                #### if never updated => force an update
                # if ( i==a[0] and j == a[1] and best_mask.size() == 0):
                #     best_mask = mask
                #     best_r = rank
                    
                mask[i][j] = -1 * mask[i][j]


    return masks, best_mask, best_r
 
        
def recherche_tabou(matrix, r):
    sqrt_matrix=np.zeros(matrix.shape)
    sqrt_matrix=np.sqrt(matrix)
    
    masks = {}              # dictionnaire hashé de tout les masks déjà testé
    
    mask=np.ones(matrix.shape)
    rank = matrix.shape[0]
    
    i = 0
    matrices, best_mask, best_rank = voisinage(masks, sqrt_matrix, mask, rank)
            
    #while i < matrix.shape[0] * matrix.shape[1]:
    while i < 2:
        if best_rank >= r:
            print(i)
            matrices, best_mask, best_rank = voisinage(masks, sqrt_matrix, best_mask, best_rank)
            i+=1
    print(best_mask)
    return best_mask, best_rank




     
#%%

matrix = np.array([[9, 4],[16,9]])

matrix = read_matrix("exempleslide_matrice.txt")



best_mask, best_r = recherche_tabou(matrix, np.shape(matrix)[0])
print(f"best mask \n {best_mask}")
print(f"best rank {best_r}")