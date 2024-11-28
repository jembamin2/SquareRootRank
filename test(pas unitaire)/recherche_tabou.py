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


def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        n, m =map(int,fin.readline().strip().split())
        
        tmp=fin.readline().strip().split()
        while tmp:
            matrix.append(list(map(float,tmp)))
            tmp=fin.readline().strip().split()
    return np.array(matrix)

def voisinage(matrices, sqrt_matrix, mask, r, a):
    best_mask = mask
    best_r = r
    for i in range(a[0]):
        for j in range(a[1]):
            mask[i][j] = -1 * mask[i][j]
            matrix_hash = hash_matrix(mask)

            # Vérifier si la matrice existe déjà dans le dictionnaire via son hash
            if matrix_hash not in matrices:
                print("new mask")
                matrices[matrix_hash] = mask  # Stocker la matrice seulement si elle est nouvelle
            
                U,S,Vh = np.linalg.svd(sqrt_matrix*mask, full_matrices= False)
                rank = rankof(S)
                S = np.diag(S)
                
                print(S)
                #print(f"mask \n {mask}")
                #print(f"rank {rank}")
                #print(((U@S)@Vh)**2)
                
                if np.allclose((((U@S)@Vh))**2, sqrt_matrix**2, atol=1e-10):
                    print("yes")
                    if rank <= r:
                        print(" new rank")
                        best_mask = mask
                        best_r = rank
                        
                
                if ( i==a[0] and j == a[1] and best_mask.size() == 0):
                    best_mask = mask
                    best_r = rank
                    
                mask[i][j] = -1 * mask[i][j]
    
    
        return matrices, best_mask, best_r

def hash_matrix(matrix):
    # Convertir la matrice en une chaîne et calculer un hash
    matrix_str = str(matrix)
    return hashlib.md5(matrix_str.encode()).hexdigest()

#%% Recherche Tabou


def recherche_tabou(sqrt_matrix, r):
    a = sqrt_matrix.shape
    matrices = {}
    #best_S=np.zeros(a)
    best_mask=np.ones(a)
    best_r = a[0]
    
    i = 0
    matrices, best_mask, best_r = voisinage(matrices, sqrt_matrix, best_mask, best_r, a)
            
    if best_r >= r:
        print(i)
        matrices, best_mask, best_r = voisinage(matrices, sqrt_matrix, best_mask, best_r, a)
        i+=1

                
            
            
            
    return best_mask, best_r
            
            
#%%

matrix = np.array([[9, 4],[16,9]])

matrix = read_matrix("exempleslide_matrice.txt")
sqrt_matrix=np.zeros(matrix.shape)
sqrt_matrix=np.sqrt(matrix)


best_mask, best_r = recherche_tabou(sqrt_matrix, np.shape(sqrt_matrix)[0])
print(f"best mask \n {best_mask}")
print(f"best rank {best_r}")