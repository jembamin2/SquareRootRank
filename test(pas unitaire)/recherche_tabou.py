# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:15:47 2024

@author: damoi
"""

import numpy as np
import hashlib
import time 
import random

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
    best_mask = mask.copy()
    best_r = r
    update = False
    print(f" best rank initial {best_r}")
    print(f" best mask initial {best_mask}")
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            mask[i][j] = -1 * mask[i][j]
            mask_hash = hash_matrix(mask)

            # Vérifier si la matrice existe déjà dans le dictionnaire via son hash
            if mask_hash not in masks:
                masks[mask_hash] = mask  # Stocker la matrice seulement si elle est nouvelle
                print("new mask")
                U,S,Vh = np.linalg.svd(matrix*mask, full_matrices= False)
                rank = rankof(S)
                S = np.diag(S)
                
                
                if np.allclose((((U@S)@Vh))**2, matrix**2, atol=1e-10):
                    if rank <= r:
                        # print(f" new rank {rank}")
                        # print(f" new mask {mask}")
                        best_mask = mask.copy()
                        best_r = rank
                        update = True
                        print("update")
                    
                    mask[i][j] = -1 * mask[i][j]    # fix me!!!!! identation
                
    if not update:
        best_mask = mask.copy()
        U,S,Vh = np.linalg.svd(matrix*mask, full_matrices= False)
        best_r = rankof(S)        
        print("force")

    return masks, best_mask, best_r
 
        
def swap(mask,index,shape):
    mask[index%shape[0],index//shape[0]]=-mask[index%shape[0],index//shape[0]]
    return mask

def recherche_tabou(matrix, r):
    sqrt_matrix=np.zeros(matrix.shape)
    sqrt_matrix=np.sqrt(matrix)
    
    masks = {}              # dictionnaire hashé de tout les masks déjà testé
    
    mask=np.ones(matrix.shape)
    mask=swap(mask,random.randint(0,mask.shape[0]*mask.shape[1]-1),matrix.shape)

    rank = matrix.shape[0]
    
    i = 0
    start = time.time()
    matrices, best_mask, best_rank = voisinage(masks, sqrt_matrix, mask, rank)
    stop = time.time()

    # if best_rank >= r:
    while stop-start <= 120 :
        print(i)
        matrices, best_mask, best_rank = voisinage(masks, sqrt_matrix, best_mask, best_rank)
        stop = time.time()
        i+=1
    print(best_mask)
    return best_mask, best_rank




     
#%%

matrix = np.array([[9, 4],[16,9]])

matrix = read_matrix("exempleslide_matrice.txt")



best_mask, best_r = recherche_tabou(matrix, np.shape(matrix)[0])
print(f"best mask \n {best_mask}")
print(f"best rank {best_r}")