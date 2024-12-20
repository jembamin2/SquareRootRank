import numpy as np
import math
import random
import hashlib
import numpy as np
import random
import hashlib

#%%
def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        n, m =map(int,fin.readline().strip().split())

        while True:
            tmp=fin.readline().strip().split(",")
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

def swap(mask,index,shape):
    mask[index%shape[0],index//shape[0]]=-mask[index%shape[0],index//shape[0]]
    return mask

def count_nonzero(array, tol=10**-1):
    count=0
    for item in array:
        if item>tol:
            count+=1
    return count

# Fonction pour générer un hash unique d'une matrice
def hash_matrix(matrix):
    # Convertir la matrice en une chaîne et calculer un hash
    matrix_str = str(matrix)
    return hashlib.md5(matrix_str.encode()).hexdigest()

# Fonction pour générer une matrice de carrés parfaits aléatoires
def generate_random_square_matrix(m, n):
    # Calculer combien d'éléments sont nécessaires
    num_elements = m * n
    
    # Générer k aléatoires pour obtenir des carrés parfaits
    random_k_values = random.choices(range(1, 15), k=num_elements)
    
    # Calculer les carrés parfaits
    squares = [k**2 for k in random_k_values]
    
    # Remplir la matrice avec ces carrés parfaits
    matrix = np.array(squares).reshape(m, n)
    
    return matrix




#%%
matrix = read_matrix("input.txt")


def generate_random_square_matrix(m, n):
    # Calculer combien d'éléments sont nécessaires
    num_elements = m * n
    
    # Générer k aléatoires pour obtenir des carrés parfaits
    # Ici, on choisit des k entre 1 et 100 pour les carrés parfaits
    random_k_values = random.choices(range(1, 15), k=num_elements)
    
    # Calculer les carrés parfaits
    squares = [k**2 for k in random_k_values]
    
    # Remplir la matrice avec ces carrés parfaits
    matrix = np.array(squares).reshape(m, n)
    
    return matrix

# Exemple d'utilisation
m, n = 10,15
matrix = generate_random_square_matrix(m, n)
print(matrix)



a=matrix.shape

mask = setup_mask(a)

sqrt_matrix=setup_sqrt_matrix(matrix)

matrices = {}

#%%
min_rank=10
best_S=np.zeros(a)
best_mask=np.zeros(a)
for _ in range(10000):
    mask=swap(mask,random.randint(0,mask.shape[0]*mask.shape[1]-1),a)
    matrix_hash = hash_matrix(mask)

    # Vérifier si la matrice existe déjà dans le dictionnaire via son hash
    if matrix_hash not in matrices:
        matrices[matrix_hash] = matrix  # Stocker la matrice seulement si elle est nouvelle
        print("Nouvelle matrice ajoutée.")
        S=np.linalg.svd(sqrt_matrix*mask)[1]
        rank=count_nonzero(S)
    else:
        print("Matrice déjà existante.")
        
    if rank<min_rank:
        min_rank=rank
        best_S=S
        best_mask=mask

U,S,Vh=np.linalg.svd(sqrt_matrix*mask)
S = np.diag(S)

padded_matrix = np.zeros((m, n))  # Create an m x n zero matrix
# Copy the diagonal matrix into the top-left corner of the padded matrix
rows, cols = S.shape
padded_matrix[:rows, :cols] = S

print(((U*S)@Vh)**2)
print(min_rank)
print(best_S)
print(best_mask)

if np.allclose((((U*S)@Vh)**2), matrix, atol=1e-8):
    print('True')
else:
    print('False')

