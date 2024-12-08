import numpy as np
from scipy.linalg import circulant
import random
import hashlib
import time
from scipy.linalg import circulant
import matplotlib.pyplot as plt

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

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

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]          # outputs: objectif1=rang, objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M,P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj(M,P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:        #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

def count_nonzero(array, tol=10**-10):
    count=0
    for item in array:
        if item>tol:
            count+=1
    return count


def hash_matrix(matrix):
    # Convertir la matrice en une chaîne et calculer un hash
    matrix_str = str(matrix)
    return hashlib.md5(matrix_str.encode()).hexdigest()

def generate_initial_solution(M, percentage=20):
    """
    Génère une solution initiale heuristique basée sur les plus faibles valeurs de M.
   
    :param M: Matrice d'entrée.
    :param percentage: Pourcentage des indices à inverser.
    :return: Solution initiale s0.
    """
    s0 = np.ones(M.shape)  # Solution initiale avec toutes les entrées à 1
    # n, m = M.shape
    # num_to_invert = max(1, (percentage * n * m) // 100)  # Nombre d'indices à inverser

    # # Identifier les indices des valeurs les plus faibles dans M
    # flat_indices = np.argsort(M.flatten())[:num_to_invert]  # Indices triés par valeur croissante
    # for idx in flat_indices:
    #     i, j = divmod(idx, m)  # Convertir l'indice 1D en indices 2D
    #     s0[i, j] = -1  # Inverser la valeur (1 -> -1)

    return s0


def diversification_par_region(s, M, threshold=1e-2, percentage=30):
    """
    Diversifie en se concentrant sur les régions de la matrice les plus problématiques
    ou prometteuses, en fonction des valeurs singulières ou des gradients locaux.
   
    :param s: Solution actuelle (matrice).
    :param M: Matrice d'entrée.
    :param threshold: Seuil pour identifier des régions problématiques (valeurs singulières faibles ou gradients).
    :param percentage: Pourcentage des éléments à modifier dans la matrice.
    :return: Nouvelle solution diversifiée.
    """
    # Calcul des valeurs singulières de la solution actuelle
    rank, singular_values = fobj(M, s)
   
    # Identifier les régions de la matrice avec des valeurs singulières faibles
    problematic_regions = np.abs(singular_values) < threshold  # Par exemple, des valeurs singulières faibles
   
    # Identifier les régions avec un grand gradient (différences importantes entre voisins)
    gradients = np.abs(np.gradient(s))  # Calcul des gradients pour la matrice
    problematic_regions_gradients = gradients > np.mean(gradients)
   
    # Créer un masque de régions problématiques (valeurs singulières faibles ou gradients élevés)
    problem_mask = np.logical_or(problematic_regions, problematic_regions_gradients)

    # Sélectionner un sous-ensemble des éléments problématiques à modifier
    n, m = s.shape
    num_to_change = max(1, (percentage * n * m) // 100)  # Nombre d'éléments à modifier
    indices_problematiques = np.argwhere(problem_mask)  # Trouver les indices des régions problématiques
   
    # Si trop peu d'éléments problématiques, élargir la recherche dans la matrice
    if len(indices_problematiques) < num_to_change:
        # Ajouter des éléments aléatoires à partir du reste de la matrice
        indices_restants = np.argwhere(np.logical_not(problem_mask))
        indices_problematiques = np.concatenate([indices_problematiques, indices_restants[:num_to_change - len(indices_problematiques)]], axis=0)
   
    # Sélectionner des indices au hasard parmi les régions problématiques
    random.shuffle(indices_problematiques)
    indices_to_flip = indices_problematiques[:num_to_change]
   
    # Créer une copie de la solution et modifier les éléments choisis
    s_prime = s.copy()
    for idx in indices_to_flip:
        i, j = idx[0], idx[1]  # Convertir l'indice en i, j
        s_prime[i, j] *= -1  # Inverser la valeur (1 ↔ -1)
   
    return s_prime


def N(s, percentage, M,nbr_voisins):
    """
    Génère une meilleure solution dans le voisinage de `s`.

    :param s: Solution courante (matrice).
    :param percentage: Pourcentage d'éléments à modifier.
    :param M: Matrice d'entrée.
    :return: Meilleure solution dans le voisinage de `s`.
    """
    s_prim = s.copy()
    n, m = s.shape
    num_to_change = max(1, (percentage * n * m) // 100)  # Nombre d'éléments à modifier
    indices = random.sample(range(n * m), num_to_change)
   
    for _ in range(nbr_voisins):
        for idx in indices:
            x = s.copy()
            i, j = divmod(idx, m)  # Convertir l'indice 1D en indices 2D
            x[i, j] *= -1  # Inverser 1 ↔ -1
            if compareP1betterthanP2(M, x, s_prim):  # Comparer avec la solution courante
                s_prim = x

    return s_prim

def recherche_tabou(M, k, percentage, max_iter,nbr_voisins,max_stagnation_counter_s,max_stagnation_counter_rank,iteration):
    """
    Algorithme de recherche tabou pour minimiser le rang de la matrice M avec une liste tabou utilisant le hash des matrices.
    :param M: Matrice d'entrée (numpy array).
    :param k: Longueur de la liste tabou.
    :param percentage: Pourcentage de valeurs à modifier pour chaque voisin.
    :param max_iter: Nombre maximal d'itérations.
    :return: Le meilleur masque trouvé.
    """
    # Initialisation
    n, m = M.shape
    s0 = generate_initial_solution(M,percentage = 30)
    s = s0.copy()
    s_opt = s0.copy()
    best_rank, _ = fobj(M, s0)  # Évaluer la solution initiale
    tabou_hashes = set()  # Ensemble pour stocker les hash des solutions taboues

    i = 0
    stagnation_counter_s = 0
    stagnation_counter_rank=0
    itera=[]
    rank_ =[]

    while i < max_iter:
        # Générer la meilleure solution voisine
        s_prim = N(s, percentage, M,nbr_voisins)
        hash_s_prim = hash_matrix(s_prim)  # Calculer le hash de la solution

        # Vérifier si `s_prim` est taboue
        if hash_s_prim not in tabou_hashes:
            rank, _ = fobj(M, s_prim)
            # Mise à jour de la meilleure solution globale
            if rank <= best_rank:
                s_opt = s_prim
                best_rank = rank
                # Ajouter le hash de `s_prim` à la liste tabou
                tabou_hashes.add(hash_s_prim)
                stagnation_counter_s = 0
                stagnation_counter_rank = 0
            else :
                stagnation_counter_rank+=1
            if len(tabou_hashes) > k:
                tabou_hashes.pop()  # Supprimer le hash le plus ancien (FIFO)
            # Mise à jour de la solution courante
            s = s_prim
        else :
            #print('deja dans liste tabou')
            stagnation_counter_s+=1
        if stagnation_counter_s >=max_stagnation_counter_s or stagnation_counter_rank >=max_stagnation_counter_rank:
            #print('DIVERSIFICATION ____________________________________')
            s=diversification_par_region(s_opt, M) #diversificaiton
            stagnation_counter_s = 0
            stagnation_counter_rank = 0
        itera.append(i)
        rank_.append(best_rank)
        i += 1  # Augmenter le compteur d'itérations
        #print(f"best rank {best_rank} x {stagnation_counter_rank}")
        #print(f"best mask {hash_matrix(s_opt)}& x {stagnation_counter_s}")
        if (stagnation_counter_s >=satgnation_max or stagnation_counter_rank >=satgnation_max):
            break
           

    plt.plot(itera,rank_,label = f'itération{iteration}')
    plt.title(f'{n},{m}')
    plt.legend()
    #plt.show()

    return s_opt


#%% Exemple d'utilisation avec matrice donnée

for i in range(20) :
    n=40
    start_time = time.time()
    matrix = matrices1_ledm(n)
    matrix_sqrt = np.sqrt(np.abs(matrix))  # Racine de chaque élément
    m=20
    if n <=8: #OKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
        k = 50 #longueur de la liste tabou
        It = (n-2)/0.3 #nombre d'itrations
        percentage = 30 #pourcentage d'index changé lors du calcul du voisinage
        nbr_voisins = 50
        max_stagnation_counter_s = 3
        max_stagnation_counter_rank = 3
        satgnation_max = 2
        
    if n <=12: #OKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
        k = 50 #longueur de la liste tabou
        It = (n-3)/0.3 #nombre d'itrations
        percentage = 30 #pourcentage d'index changé lors du calcul du voisinage
        nbr_voisins = 50
        max_stagnation_counter_s = 4
        max_stagnation_counter_rank = 4
        satgnation_max = 3
    elif n<=17 :    
        k = 20 #longueur de la liste tabou
        It = (n-3)/0.3 #nombre d'itrations
        percentage = 50 #pourcentage d'index changé lors du calcul du voisinage
        nbr_voisins = 30
        max_stagnation_counter_s = 5
        max_stagnation_counter_rank = 5
        satgnation_max = 4
    elif n<=30:
        k = 1000 #longueur de la liste tabou
        It = 500 #nombre d'itrations
        percentage = 40 #pourcentage d'index changé lors du calcul du voisinage
        nbr_voisins = 25
        max_stagnation_counter_s = 5
        max_stagnation_counter_rank = 5
        satgnation_max = 4
    elif n<=40:
        k = 1000 #longueur de la liste tabou
        It = 300 #nombre d'itrations
        percentage = 40 #pourcentage d'index changé lors du calcul du voisinage
        nbr_voisins = 20
        max_stagnation_counter_s = 5
        max_stagnation_counter_rank = 5
        satgnation_max = 4
    # elif n<=50:
    #     k = 40 #longueur de la liste tabou
    #     It = 200 #nombre d'itrations
    #     percentage = 50 #pourcentage d'index changé lors du calcul du voisinage
    #     nbr_voisins = 5
    #     max_stagnation_counter_s = 3
    #     max_stagnation_counter_rank = 10
    # else :
    #     k = 150 #longueur de la liste tabou
    #     It = 50 #nombre d'itrations
    #     percentage = 50 #pourcentage d'index changé lors du calcul du voisinage
    #     nbr_voisins = 3
    #     max_stagnation_counter_s = 3
    #     max_stagnation_counter_rank = 3
   
    best=dict()
   
    for i in range(m):
        best_mask = recherche_tabou(matrix, k=k, percentage = percentage, max_iter=It,nbr_voisins = nbr_voisins,max_stagnation_counter_s = max_stagnation_counter_s,max_stagnation_counter_rank = max_stagnation_counter_rank,iteration = i)
        U,S,Vh = np.linalg.svd(matrix_sqrt*best_mask, full_matrices= False)
        best_rank = count_nonzero(S)
        best[best_rank] = best_mask
    best_rank = min(best)
    best_mask = best[best_rank]
    #print(f"best mask \n {best_mask}")
    print(f"best rank {n}->{best_rank}")
    #print(S)
     
     
    if not np.allclose((((U*S)@Vh)**2), matrix, atol=1e-8):
          print('False')
    else :
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Temps d'exécution : {execution_time:.2f} secondes")
       