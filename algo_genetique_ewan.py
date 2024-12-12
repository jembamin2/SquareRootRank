import numpy as np
import math
import random
import hashlib
import numpy as np
import random
import hashlib
from tqdm import tqdm

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
        mask = np.random.choice([-1,1,1,1,1,1,1,1,1,1,1, 1], size=(m,n ))
        pop.append(mask)
    return pop

def fobj(M,P,tol=1e-14):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False) # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                # indices des valeurs > tolérance donnée
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]       # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle

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

# Exemple d'utilisation
n = 16
m=n
matrix = matrices1_ledm( n)
sqrt_matrix=setup_sqrt_matrix(matrix)



#%%
tol=1e-14
min_rank=min(m, n)
best_mask=np.zeros((m, n))

iterations=10000
nb_pop=200
nb_parents=150
nb_parents_random=50
nb_mut=3
pop=init_genetique(m,n,nb_pop)
perf=np.zeros(nb_pop)
histo_perf=np.zeros(iterations)
counter=0
mutations=nb_mut
tmp=0
type_mut=0
with tqdm(total=iterations) as pbar:
    for i in range(iterations):
        pop,parents=selection_reproduction2(matrix, pop, nb_parents, nb_parents_random)
        mask_croisement=np.random.choice([True,False],m,n)
        childs=croisement(parents,mask_croisement)
        match type_mut:
            case 0:
                infos=f"random mutations avec {mutations} mutations"
                childs=rand_mutation(childs, mutations)
            case 1:
                infos=f"recherche locale avec {mutations} mutations"
                childs=rech_loc_mut(0,childs, mutations)
        a=nb_pop-(nb_parents+nb_parents_random)//2
        pop=pop[:a]+childs
        tmp=fobj(matrix,pop[0])
        histo_perf[i]=tmp[1]
        if histo_perf[i]==histo_perf[i-1]:
            counter+=1
            if (counter>5):
             #   mutations=round(math.log(counter*5))*nb_mut
                type_mut=0
        else:
            #type_mut=1
            counter=0
            mutations=nb_mut
        pbar.write(infos)
        pbar.set_postfix({"rang" : tmp[0],"smallest singular value": tmp[1]})
        pbar.update(1)
    
pop = sorted(pop, key=lambda ind: fobj(matrix, ind))
best_mask=pop[0]

U,S,Vh=np.linalg.svd(sqrt_matrix*best_mask,full_matrices=False)
S = np.diag(S)

S[S<tol]=0

if np.allclose((((U@S)@Vh)**2), matrix, atol=tol):
    print('True')
else:
    print('False')