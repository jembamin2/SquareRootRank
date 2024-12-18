import numpy as np
import math
import random
import hashlib
import numpy as np
import random
import hashlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

#%%
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

def count_nonzero(array, tol=10**-8):
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

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

def init_genetique(m,n, nb_pop):
    pop = []
    for _ in range(nb_pop):
        mask = np.random.choice([ 1], size=(n,m ))
        pop.append(mask)
    return pop

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]  

def selection_reproduction(pop,perf,nb_parents):
    return list(np.array(pop)[np.argsort(perf)])[:nb_parents]

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
        child = np.where(mask_croisement,parents[index[0]],parents[index[1]])
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
   


def rech_loc_mut2(compteur,childs, nb_mut):
    a=childs[0].shape
    T=random.randint(0, a[0]*a[1]-1)
    n=a[0]
    m=a[1]
    while compteur<nb_mut:
        for child in childs:
            best_neighbor=sum(list(fobj(matrix,child)))
            best_i=0
            for i in range(T):
                i, j = random.randint(0, n - 1), random.randint(0, m - 1)
                block_size = 2
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, m)
                
                new_mask = child
                new_mask[i:i_end, j:j_end] *= -1
                
                new_rank, singular = fobj(matrix, new_mask)
                                    
                act_neighbor=sum(list(fobj(matrix,new_mask)))
                if act_neighbor<best_neighbor:
                    best_neighbor=act_neighbor
                    best_i=i
            child=swap(child,best_i,a)
        compteur+=1
        rech_loc_mut(compteur,childs,nb_mut)

    return childs


def rech_loc_mut(compteur,childs, nb_mut):
    a=childs[0].shape
    while compteur<nb_mut:
        for child in childs:
            best_neighbor=sum(list(fobj(matrix,child)))
            best_i=0
            for i in range(a[0]*a[1]-1):
                act_neighbor=sum(list(fobj(matrix,swap(child,i,a))))
                if act_neighbor<best_neighbor:
                    best_neighbor=act_neighbor
                    best_i=i
            child=swap(child,best_i,a)
        compteur+=1
        rech_loc_mut(compteur,childs,nb_mut)

    return childs

from scipy.linalg import circulant
def matrices2_slackngon(n):
  M  = circulant(np.cos(np.pi/n)-np.cos(np.pi/n + 2*np.pi*np.arange(0,n,1)/n))
  M /= M[0,2]
  M  = np.maximum(M,0)
  for i in range(n):
    M[i,i] = 0
    if i<n-1:
      M[i,i+1] = 0
    else:
      M[i,0] = 0
  return M

def save_matrix(M, P):
    # Calculate singular values
    sing_values = np.linalg.svd(P * M, compute_uv=False)
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]

    # Current rank and largest singular value
    current_rank = len(ind_nonzero)
    smallest_singular_value = sing_values[ind_nonzero[-1]]

    # Check for existing files
    existing_files = [f for f in os.listdir('.') if f.startswith("output_rank")]
    # print(len(existing_files), current_rank)
    # random_component = random.randint(1, 999999)
    file_name = False

    for file in existing_files:
        try:
            # Extract rank from the filename
            rank_in_file = int(file.split('_')[1].split('.')[0][4:])  # Extract "5" from "output_rank5..."
            # print(rank_in_file, " vs ", current_rank)
            if rank_in_file > current_rank:
                # print("here")
                file_name = f"output_rank{current_rank}.txt"

                # Save the new file
                with open(file_name, 'w') as fout:
                    np.savetxt(fout, P, fmt='%.0f', delimiter=' ')
                    for i in ind_nonzero:
                        fout.write(f'{sing_values[i]}\n')
                
                break

            # If the file has the same rank, compare singular values
            elif rank_in_file == current_rank:
                # print(rank_in_file, " vs ", current_rank)
                # print("heerreeee")
                with open(file, 'r') as f:
                    lines = f.readlines()
                    # Read the last singular value in the file
                    last_saved_singular_value = float(lines[-1].strip())
                    # print(f"Last saved singular value: {last_saved_singular_value}")
                    # print(f"Smallest singular value: {smallest_singular_value}")
                    if last_saved_singular_value > smallest_singular_value:
                        file_name = f"output_rank{current_rank}.txt"

                        # Save the new file
                        with open(file_name, 'w') as fout:
                            np.savetxt(fout, P, fmt='%.0f', delimiter=' ')
                            for i in ind_nonzero:
                                fout.write(f'{sing_values[i]}\n')
                break

        except (IndexError, ValueError, FileNotFoundError):
            continue

    if existing_files == []:
        # print("hereAMIE")
        file_name = f"output_rank{current_rank}.txt"

        # Save the new file
        with open(file_name, 'w') as fout:
            np.savetxt(fout, P, fmt='%.0f', delimiter=' ')
            for i in ind_nonzero:
                fout.write(f'{sing_values[i]}\n')
    # if file_name:
        # print(f"Matrix saved to {file_name}")

#matrix = matrices1_ledm(35)
#matrix=read_matrix("test(pas unitaire)/slack7gon_matrice.txt")
#matrix=matrices2_slackngon(15)
matrix=read_matrix("file.txt")
n,m=matrix.shape
sqrt_matrix=setup_sqrt_matrix(matrix)



#%%
tol=1e-14
min_rank=min(m, n)
best_mask=np.zeros((m, n))

iterations=1000
rank_evolution=np.zeros(iterations)

nb_pop=200
nb_parents=50
nb_parents_random=150
nb_mut=1
pop=init_genetique(m,n,nb_pop)
perf=np.zeros(nb_pop)
histo_perf=np.zeros(iterations)
counter=0
mutations=nb_mut
tmp=0
type_mut=0
with tqdm(total=iterations) as pbar:
    for i in range(iterations):
        tmp=fobj(matrix,pop[0])
        histo_perf[i]=tmp[1]
        if histo_perf[i]==histo_perf[i-1]:
            counter+=1
            if (counter>20 and counter%20==0):
                mutations=3
                #mutations=round(math.log(counter*5))*nb_mut
                #mutations=round(counter/5)*nb_mut
                type_mut=0
            if counter>30 and counter%30==0:
                mutations=1
                
            if counter==1001:
                 mutations=200
                 print('diversification magueule')
           
            
        else:
            save_matrix(sqrt_matrix, pop[0])
            type_mut=0
            counter=0
            mutations=nb_mut
        
        pop,parents=selection_reproduction2(matrix, pop, nb_parents, nb_parents_random)
        mask_croisement=np.random.choice([True,False],m,n)
        childs=croisement(parents,mask_croisement)
        match type_mut:
            case 0:
                infos=f"random mutations avec {mutations} mutations"
                childs=rand_mutation(childs, mutations)
            case 1:
                infos=f"recherche locale avec {mutations} mutations"
                childs=rech_loc_mut2(0,childs, mutations)
        a=nb_pop-(nb_parents+nb_parents_random)//2
        pop=pop[:a]+childs
        #pbar.write(infos)
        pbar.set_postfix({"rang" : tmp[0],"smallest singular value": tmp[1]})
        pbar.update(1)
        rank_evolution[i]=tmp[0]
    
pop = sorted(pop, key=lambda ind: fobj(matrix, ind))
best_mask=pop[0]

U,S,Vh=np.linalg.svd(sqrt_matrix*best_mask,full_matrices=False)
S = np.diag(S)

S[S<tol]=0

if np.allclose((((U@S)@Vh)**2), matrix, atol=tol):
    print('True')
else:
    print('False')
#%%
test=((U@S)@Vh)**2





#%%

plt.plot(rank_evolution)
plt.title("Genetic Rank Evolution")
plt.xlabel("iterations")
plt.ylabel("rank")
