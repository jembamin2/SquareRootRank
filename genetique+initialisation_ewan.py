import numpy as np
import math
import random
import hashlib
import numpy as np
import random
import hashlib
from tqdm import tqdm

#%%
def genetique(matrix, initial_parent,iterations ):
    best_mask=np.zeros((m, n))
    
    nb_pop=200
    nb_parents=50
    nb_parents_random=150
    nb_mut=1
    pop=init_genetique2(initial_parent,nb_pop)
    histo_perf=np.zeros(iterations)
    counter=0
    mutations=nb_mut
    tmp=0
    with tqdm(total=iterations) as pbar:
        for i in range(iterations):
            tmp=fobj(matrix,pop[0])
            histo_perf[i]=tmp[1]
            if histo_perf[i]==histo_perf[i-1]:
                counter+=1
                if (counter>20):
                    mutations=round(counter/5)*nb_mut
                    if(counter>50):
                        break
                else:
                    counter=0
                    mutations=nb_mut
            pop,parents=selection_reproduction2(matrix, pop, nb_parents, nb_parents_random)
            mask_croisement=np.random.choice([True,False],initial_parent.shape)
            childs=croisement(parents,mask_croisement)
            childs=rand_mutation(childs, mutations)
            a=nb_pop-(nb_parents+nb_parents_random)//2
            pop=pop[:a]+childs
            test=fobj(matrix,pop[0])
            pbar.set_postfix({"rang" : test[0],"smallest singular value": test[1]})
            pbar.update(1)
    pop = sorted(pop, key=lambda ind: fobj(matrix, ind))
    return pop[0]
    
        
        
def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        n, m =map(int,fin.readline().strip().split())

        while True:
            tmp=fin.readline().strip().split(" ")
            if tmp==[""]:
                break
            matrix.append(list(map(float,tmp)))
    return np.array(matrix)



def setup_sqrt_matrix(matrix):
    sqrt_matrix=np.zeros(matrix.shape)
    sqrt_matrix=np.sqrt(matrix)
    return sqrt_matrix

def swap(mask,index,shape):
    mask[index%shape[0],index//shape[0]]=-mask[index%shape[0],index//shape[0]]
    return mask

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

def init_genetique2(initial_parent, nb_pop):
    pop = []
    for _ in range(nb_pop):
        mask = initial_parent
        pop.append(mask)
    return pop

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]  


def selection_reproduction2(matrix,pop,nb_parents,percentage):
    pop = sorted(pop, key=lambda ind: fobj(matrix, ind))
    select=pop[:nb_parents]
    rand_indx=np.random.choice(range(nb_parents,len(pop)),percentage)
    rand=[]
    for indx in rand_indx:
        rand.append(pop[indx])
    return pop,select+rand

def croisement(parents,mask_croisement):
    temp_parents=parents.copy()
    childs=[]
    for i in range(len(parents)//2):
        index = np.random.choice(range(len(temp_parents)-1), size=2)
        child = np.where(mask_croisement,temp_parents[index[0]],parents[index[1]])
        childs.append(child)
        temp_parents.pop(index[0])
        temp_parents.pop(index[1])
    return childs

def rand_mutation(childs, nb_mut):
    a=childs[0].shape
    for child in childs:
        for i in range(nb_mut):
            child=swap(child,random.randint(0,child.shape[0]*child.shape[1]-1),a)
    return childs




#%%
        
# Exemple d'utilisation
n = 120
m=n
#initial_matrix = matrices1_ledm( n)
initial_matrix=read_matrix("test(pas unitaire)/correl5_matrice.txt")
sqrt_matrix=setup_sqrt_matrix(initial_matrix)



initial_block_size=8
scaling_factor=2
n = initial_matrix.shape[0]  # Size of the full matrix

# Initialize a global mask with all ones
global_mask = np.ones_like(initial_matrix, dtype=int)
# Start with the smallest block size and progressively increase
block_size = initial_block_size
while block_size <= n:
    num_blocks = int(n // block_size)  # Number of blocks along each dimension
    # Refine the mask using the current block size
    for i in range(num_blocks):
        for j in range(num_blocks):
            # Extract the block
            block = initial_matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            
            # Extract the corresponding portion of the global mask
            block_mask_initial = global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            
            # Apply tabu search to optimize the block
            block_mask = genetique(block, block_mask_initial, 300)
            
            # Update the corresponding portion of the global mask
            global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block_mask
    
    
    # Increase the block size for the next iteration
    block_size *= scaling_factor


block_mask = genetique(initial_matrix, global_mask, 20000)
