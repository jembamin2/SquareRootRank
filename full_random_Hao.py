import numpy as np
import math
import random

from numpy.linalg import svd


def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        while True:
            tmp=fin.readline().rsplit(",")
            if tmp==[""]:
                break
            matrix.append(list(map(int,tmp)))
    return np.array(matrix)


def setup_mask(shape):
    mask=np.ones(shape)
    return mask

def setup_sqrt_matrix(matrix):
    sqrt_matrix=np.zeros(matrix.shape)
    sqrt_matrix=np.sqrt(matrix)
    return sqrt_matrix

def swap(mask, shape):
    # Choisir une position aléatoire (ligne, colonne)
    row = random.randint(0, shape[0] - 1)
    col = random.randint(0, shape[1] - 1)
    # Inverser le signe de l'élément correspondant
    mask[row, col] = -mask[row, col]
    return mask


def count_nonzero(array, tol=10**-1):
    count=0
    for item in array:
        if item>tol:
            count+=1
    return count

def evaluate_matrix(matrix, tol=1e-10):
    # Décomposer la matrice en valeurs singulières
    _, singular_values, _ = svd(matrix)
    # Calculer le rang (nombre de valeurs singulières > tolérance)
    rank = np.sum(singular_values > tol)
    # Obtenir la plus petite valeur singulière non nulle
    smallest_singular = singular_values[-1]
    # Filtrer toutes les valeurs singulières supérieures à la tolérance
    significant_singular_values = singular_values[singular_values > tol]
    return rank, smallest_singular, significant_singular_values



def optimize_matrix(sqrt_matrix, mask, num_iterations):
    best_mask = mask.copy()
    best_rank = float('inf')  # Initialiser avec un rang très élevé
    best_singular = float('inf')  # Initialiser avec une valeur singulière très élevée
    best_significant_singular_values = []  # Liste pour stocker les meilleures valeurs singulières

    for iteration in range(num_iterations):
        # Générer un masque aléatoire en inversant un élément au hasard
        new_mask = swap(mask, mask.shape)
        
        # Appliquer le masque à la matrice
        test_matrix = sqrt_matrix * new_mask
        
        # Calculer le rang et les valeurs singulières significatives
        rank, smallest_singular, significant_singular_values = evaluate_matrix(test_matrix)
        
        # Mettre à jour les meilleurs résultats si nécessaire
        if (rank < best_rank) or (rank == best_rank and smallest_singular < best_singular):
            best_mask = new_mask
            best_rank = rank
            best_singular = smallest_singular
            best_significant_singular_values = significant_singular_values
        
        # Affichage des résultats intermédiaires
        print(f"{iteration}: rank={rank} ss={significant_singular_values[-1]}")

    # Retourner les meilleurs résultats trouvés
    return best_mask, best_rank, best_singular, best_significant_singular_values


#%%

matrix = read_matrix("input.txt")

a=matrix.shape
mask = setup_mask(a)
sqrt_matrix=setup_sqrt_matrix(matrix)

num_iterations = 100000
best_mask,best_rank,best_singular,best_significant_singular_values=optimize_matrix(sqrt_matrix, mask, num_iterations)

print("Best rank:", best_rank)
print("Smallest singular value:", best_significant_singular_values[-1])

