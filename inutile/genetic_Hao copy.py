import opti_combi_projet_pythoncode_texte_v2 as opti
import numpy as np
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

#%%

def read_matrix(input_file):
    with open(input_file, 'r') as fin:
        matrix = []
        r,c=map(int,fin.readline().split())
        for i in range (r):
            tmp = fin.readline().split()
            matrix.append(list(map(float, tmp)))
    return np.array(matrix)

def metaheuristic(M, 
                  sa_solutions,
                  pop_size, 
                  generations, 
                  mutation_rate, 
                  num_parents, 
                  num_children,
                  max_stagnation):
    stagnation = 0
    start=time.time()
    
    m, n = M.shape  # Dimensions de la matrice cible
    # indices = [(i, j) for i in range(m) for j in range(n)]

    
    def generate_clever_individual():
        
        # individual = np.random.choice([-1, 1], size=(m, n), p=[0.85,0.15])
        individual = np.ones((m, n))
        for i in range(m):
            for j in range(n):
                if M[i, j] < np.median(M):
                    individual[i, j] = -1
        return individual
    
    def generate_population(size, sa_solutions):
        population = [generate_clever_individual() for _ in range((size-len(sa_solutions)))]
        return sa_solutions + population

    # Fitness : Évaluation du pattern
    def fitness(individual):
        r, s = opti.fobj(M, individual)
        return r, s

    # Méthodes de sélection des parents

    def select_parents_tournament(population, num_parents):
        """Sélection par tournoi avec retrait des compétiteurs choisis."""
        selected_parents = []
        for _ in range(num_parents):
            if len(population) < 2:  # S'assurer qu'il reste au moins 2 individus pour le tournoi
                raise ValueError("La population est insuffisante pour un tournoi.")
            
            competitors = random.sample(population, 2)
            competitors = sorted(competitors, key=lambda ind: fitness(ind))
            # print(f"Fitness: {fitness(competitors[0])} vs {fitness(competitors[1])}")
            
            selected_parents.append(competitors[0])  # Le meilleur gagne

            # Rechercher et retirer l'individu correspondant
            for i, individual in enumerate(population):
                if (individual == competitors[0]).all():  # Comparaison spécifique aux tableaux
                    del population[i]
                    break
        return selected_parents


    
    def select_parents_roulette(population, num_parents):
        """Sélection par roulette.""" #A MODIFIER POUR PROBA
        fitness_values = [1/(fitness(ind)[1]) for ind in population]
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        return random.choices(population, probabilities, k=num_parents)

    # Liste des méthodes de sélection des parents
    parent_selection_methods = [
        select_parents_tournament
        # ,select_parents_roulette
    ]

    # Opérateurs de croisement
    def crossover_uniform(parent1, parent2):
        mask = np.random.randint(0, 2, size=(m, n))
        return np.where(mask, parent1, parent2)
    
    def crossover_one_point(parent1, parent2):
        point = random.randint(0, m * n - 1)
        flat1, flat2 = parent1.flatten(), parent2.flatten()
        child = np.concatenate((flat1[:point], flat2[point:]))
        return child.reshape(m, n)
    
    def crossover_multi_point(parent1, parent2, num_points=3):
        points = sorted(random.sample(range(m * n), num_points))
        flat1, flat2 = parent1.flatten(), parent2.flatten()
        child = flat1.copy()
        for i, point in enumerate(points):
            if i % 2 == 0:
                child[point:] = flat2[point:]
            else:
                child[point:] = flat1[point:]
        return child.reshape(m, n)

    # Liste des méthodes de croisement
    crossover_methods = [
        crossover_uniform,
        crossover_one_point,
        crossover_multi_point
    ]
    
    # Mutation : inverser -1 en 1 et 1 en -1
    def mutate_many(individual):
        for i in range(m):
            for j in range(n):
                if random.random() < mutation_rate:
                    individual[i, j] *= -1
        return individual

    def mutate_1(individual):
        i = random.randint(0, m-1)
        j = random.randint(0, n-1)
        individual[i, j] *= -1
        return individual

    def mutate_5(individual):
        swap_indices = []
        for _ in range(5): 
            i = random.randint(0, m-1)
            j = random.randint(0, n-1)
            swap_indices.append((i, j))

        for (i, j) in swap_indices:
            individual[i, j] *= -1
        return individual

    mutate_methods = [
        mutate_many,
        mutate_1,
        mutate_5
    ]

    # Méthode pour sélectionner la nouvelle génération
    def generation_elitism(parents, children, population):
            combined = parents + children
            return sorted(combined, key=lambda ind: fitness(ind))[:pop_size]
    
    def generation_combined(parents, children, population):
            return sorted(population + children, key=lambda ind: fitness(ind))[:pop_size]
    
    def generation_all(parents, children, population):
            return sorted(population + parents + children, key=lambda ind: fitness(ind))[:pop_size]
    
    new_generation_methods = [
        generation_elitism, 
        generation_combined,
        generation_all
    ]

    # Initialisation
    population = sorted(generate_population(pop_size, sa_solutions), key=lambda ind: fitness(ind))
    bestPattern = population[0]
    best_fitness = fitness(bestPattern)
    bestPatternB = population[0]
    best_fitnessB = fitness(bestPatternB)

    # Algorithme génétique
    for gen in (range(generations)):
        # Choisir aléatoirement une méthode de sélection des parents
        parent_selection_method = random.choice(parent_selection_methods)
        parents = parent_selection_method(population, num_parents)
        
        # Génération des enfants
        children = []
        for _ in range(num_children):
            parent1, parent2 = random.sample(parents, 2)
            
            # Choisir aléatoirement une méthode de croisement
            crossover_method = random.choice(crossover_methods)
            child = crossover_method(parent1, parent2)
            
            if random.random() < mutation_rate:
                # Mutation
                mutate_method = random.choices(
                    mutate_methods, 
                    weights=[0.2, 0.2, 0.65], 
                    k=1
                )[0]
                child = mutate_method(child)
            children.append(child)
        
        # Nouvelle génération
        new_generation_method = random.choice(new_generation_methods)
        population = new_generation_method(parents, children, population)

        stagnation += 1

        # Mettre à jour le meilleur pattern
        best_in_pop = sorted(population, key=lambda ind: fitness(ind))[0]
        if opti.compareP1betterthanP2(M, best_in_pop, bestPattern):
            bestPattern = best_in_pop
            best_fitness = fitness(bestPattern)
            stagnation = 0
            if opti.compareP1betterthanP2(M, bestPattern, bestPatternB):
                bestPatternB = bestPattern
                best_fitnessB = fitness(bestPatternB)
            # print(best_fitness)

        if stagnation > max_stagnation:
            print("Stagnation")
            population = sorted(generate_population(pop_size, sa_solutions), key=lambda ind: fitness(ind))
            bestPattern = population[0]
            best_fitness = fitness(bestPattern)
            stagnation = 0

        print(f"Génération {gen+1} - Rang : {best_fitness[0]}, Petite valeur singulière : {best_fitness[1]}")
        # if best_fitness[0] == 2:
        #     break

        # if time.time()-start>300:
        #     break

    print(time.time()-start)
    return bestPatternB


# Appliquer le masque à la matrice
def apply_mask(matrix, mask):
    return matrix * mask

# Calculer le rang
def compute_rank(matrix):
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    rank = np.sum(S > 1e-10)
    return rank

def fobj(M,P=1):
    sing_values = np.linalg.svd(M*P, compute_uv=False)
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]

def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M,P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj(M,P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:        #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

def initialize_mask_from_low_rank_blocks(matrix, block_size=10):
    blocks = split_into_blocks(matrix, block_size)
    mask = np.ones(matrix.shape)

    for block, i, j in blocks:
        if np.linalg.matrix_rank(block) <= 2:  # Seuil pour les blocs à faible rang  Il faudrait print pour savoir ce que ca retourne car definition dit tol=None
            mask[i:i+block_size, j:j+block_size] = -1

    return mask

def split_into_blocks(matrix, block_size):
    """
    Divise une matrice en blocs de taille `block_size x block_size`.
    """
    n, m = matrix.shape
    blocks = []
    for i in range(0, n, block_size):
        for j in range(0, m, block_size):
            block = matrix[i:i+block_size, j:j+block_size]
            blocks.append((block, i, j))  # Garder les indices de début
    return blocks

def optimize_block_mask(block, threshold=1e-12):
    """
    Optimise le masque pour une sous-matrice donnée.
    """
    U, S, Vh = np.linalg.svd(block, full_matrices=False)
    mask = np.ones(block.shape)
    for i, singular_value in enumerate(S):
        if singular_value < threshold:
            mask[i, :] = -1  # Marquer les lignes faibles
    return mask

def reconstruct_global_mask(blocks, masks, matrix_shape):
    """
    Reconstruit le masque global à partir des masques des blocs.
    """
    global_mask = np.ones(matrix_shape)
    for (block, i, j), mask in zip(blocks, masks):
        n, m = mask.shape
        global_mask[i:i+n, j:j+m] = mask
    return global_mask

def initialize_mask_from_blocks(matrix, block_size, threshold=1e-12):
    # Diviser la matrice en blocs
    blocks = split_into_blocks(matrix, block_size)
    # print(blocks)
    
    # Optimiser le masque pour chaque bloc
    masks = [optimize_block_mask(block, threshold) for block, _, _ in blocks]
    
    # Reconstruire le masque global
    global_mask = reconstruct_global_mask(blocks, masks, matrix.shape)
    return global_mask

def generate_neighbors(current_mask, num_neighbors, num_modifications):
    """
    Generates neighbors by flipping a random set of entries in the current_mask.
    """
    n_rows, n_cols = current_mask.shape
    neighbors = []

    # Randomly pick `num_modifications` positions to flip
    total_elements = n_rows * n_cols
    all_indices = np.arange(total_elements)
    
    for _ in range(num_neighbors):
        # Create a new neighbor by copying the current mask
        neighbor = current_mask.copy()
        
        # Randomly select `num_modifications` indices to flip
        flip_indices = np.random.choice(all_indices, num_modifications, replace=False)
        
        # Flip the selected indices (invert their values)
        for idx in flip_indices:
            i, j = divmod(idx, n_cols)  # Convert flat index to 2D (i, j)
            neighbor[i, j] *= -1
        
        neighbors.append(neighbor)
    
    return neighbors

def dynamic_neigh_modulation(iteration, max_iterations, initial_neighbors, initial_modifications):
    """
    Dynamically adjusts the number of neighbors and modifications based on the iteration number.
    """
    progress = iteration / max_iterations
    num_neighbors = int(initial_neighbors * (1 - progress))  # Reduce neighbors as progress increases
    num_modifications = max(1, int(initial_modifications * (1 - progress)))  # Reduce modifications, ensure at least 1
    return num_neighbors, num_modifications



def tabu_search_with_plot(matrix, initial_mask, tot_resets, num_n, num_m, tabu_size, max_no_improve, max_iterations):
    """
    Recherche tabou pour minimiser le rang avec graphiques.
    """
    # Historique pour tous les voisinages
    all_rank_histories = []
    rank_history = []
    best_overall_rank = float('inf')
    best_overall_singular_value = float('inf')
    best_overall_mask = initial_mask.copy()
    total_resets = 0

    best_rank = best_overall_rank;
    
    tabu_set = set()

    
    while total_resets < tot_resets and best_rank > 2:  # Limite sur le nombre de voisinages explorés
        # Initialisation pour un nouveau voisinage
        current_mask = initial_mask.copy()
        best_mask = current_mask.copy()
        current_rank, current_singular_value = fobj(matrix)
        best_rank = current_rank
        best_singular_value = current_singular_value
        no_improve = 0

        rank_history = []
        print(f"=== Exploration du voisinage {total_resets + 1} ===")
        for iteration in range(max_iterations):
            num_neighbors, num_modifications = dynamic_neigh_modulation(iteration, max_iterations, num_n, num_m)
            neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)            # Générer plusieurs voisins
            # neighbors = []
            # n_rows, n_cols = current_mask.shape
            # num_neighbors = num_n
            # num_modifications = num_m
            
            if iteration == 1:
                print(f"neigh = {num_neighbors}")
                print(f"mod = {num_modifications}")
                # Générer les voisins dynamiquement
            # neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)

        
            # Évaluer tous les voisins
            evaluated_neighbors = []
            for neighbor in neighbors:
                neighbor_bytes = neighbor.tobytes()
                if neighbor_bytes not in tabu_set:
                    masked_matrix = apply_mask(matrix, neighbor)
                    neighbor_rank, neighbor_singular_value = fobj(masked_matrix)
                    evaluated_neighbors.append((neighbor_rank, neighbor_singular_value, neighbor_bytes))
            
            # Sort the neighbors by rank, then by singular value
            evaluated_neighbors.sort(key=lambda x: (x[0], x[1]))
            
            # Prendre le meilleur voisin (ou plusieurs)
            if evaluated_neighbors:
                best_neighbor_rank, best_neighbor_singular_value, best_neighbor_bytes = evaluated_neighbors[0]
                current_mask = np.frombuffer(best_neighbor_bytes, dtype=current_mask.dtype).reshape(current_mask.shape) #je comrpends pas TODO
                current_rank = best_neighbor_rank
                current_singular_value = best_neighbor_singular_value
        
                rank_history.append((best_rank, current_singular_value))

                if compareP1betterthanP2(matrix,current_mask,best_mask):
                    best_mask = current_mask.copy()
                    best_rank = current_rank
                    best_singular_value = current_singular_value
                    print(f"no improve: {iteration - no_improve}, rank = {current_rank}")
                    no_improve = 0
                else:
                    print(f"no improve: {iteration - no_improve}, rank_naze = {current_rank}")
                    no_improve += 1

                # Add to tabu set
                tabu_set.add(best_neighbor_bytes)
                if len(tabu_set) > tabu_size:
                    tabu_set.pop()
        
            if best_rank == 2:
                break
            
            # Si aucune amélioration après plusieurs itérations
            if no_improve >= max_no_improve:
                print(f"num iterations: {iteration}, rank = {best_rank}")
                break

        
            #print(f"Iteration {iteration + 1}: Current Rank = {current_rank}, Best Rank = {best_rank}")

        # Mettre à jour le meilleur résultat global
        if compareP1betterthanP2(matrix,best_mask,best_overall_mask):
            best_overall_rank = best_rank
            best_overall_singular_value = best_singular_value
            best_overall_mask = best_mask


        # Sauvegarder l'historique du voisinage
        all_rank_histories.append(rank_history)

        # Réinitialiser pour un nouveau voisinage
        total_resets += 1

    # # Tracer les rangs pour tous les voisinages
    # plt.figure(figsize=(12, 8))

    # # Parcourir chaque voisinage et tracer son historique de rangs
    # for i, rank_history in enumerate(all_rank_histories):
    #     iterations = range(len(rank_history))  # Réinitialiser les itérations pour chaque voisinage
    #     ranks = [rank for rank, _ in rank_history]  # Extraire les rangs
    #     plt.plot(iterations, ranks, label=f"Voisinage {i + 1}")  # Tracer une ligne pour chaque voisinage
    
    # # Ajouter une ligne pour le meilleur rang global
    # plt.axhline(y=best_overall_rank, color='r', linestyle='--', label="Meilleur rang global")
    
    # Ajouter des détails pour rendre le graphique plus lisible
    # plt.xlabel("Itérations (réinitialisées pour chaque voisinage)")
    # plt.ylabel("Rang")
    # plt.title("Évolution du rang pour chaque voisinage (Recherche Tabou)")
    # plt.legend()  # Ajouter une légende
    # plt.grid(True)  # Afficher la grille
    # plt.show()

    return best_overall_mask, best_overall_rank


#%%

# M = read_matrix("test(pas unitaire)/correl5_matrice.txt")
# M = read_matrix("test(pas unitaire)/slack7gon_matrice.txt")
# M = read_matrix("test(pas unitaire)/synthetic_matrice.txt")
M = read_matrix("file.txt")


# m, n = 10, 10
# M = np.random.rand(m, n)
# M = opti.matrices1_ledm(15)
sols = [M]

sol = []
sa_solutions=[]


original_matrix = M
sqrt_matrix = np.sqrt(original_matrix)
initial_mask = np.ones_like(sqrt_matrix)
in_rank, s = fobj(sqrt_matrix)

# Exemple d'application
block_size = 6  # Taille des sous-matrices

#block_mask = initialize_mask_from_low_rank_blocks(sqrt_matrix, block_size=5)
block_mask = initialize_mask_from_blocks(original_matrix, block_size)
print("Masque basé sur blocs :", block_mask)

# Appliquer la recherche tabou avec graphique
start = time.time()
for i in range(25):
    #best_mask, best_rank = tabu_search_with_plot(sqrt_matrix, initial_mask,mod_neighbors = (0.5, 0.01), mod_mod = (0.025, 0.01), tabu_size=1000, max_no_improve=2000, max_iterations=1000000)
    best_mask, best_rank = tabu_search_with_plot(
        sqrt_matrix, block_mask,
        tot_resets = 1,
        num_n = 200,
        num_m = 1,
        tabu_size=100000, 
        max_no_improve=100, 
        max_iterations=10000000)
    
    sa_solutions.append(best_mask)

print("Temps d'exécution :", time.time() - start)


start=time.time()
best_pattern = metaheuristic(
    M, 
    sa_solutions,
    pop_size=250,
    generations=1500, 
    mutation_rate=0.6, 
    num_parents=150, 
    num_children=300,
    max_stagnation=200
)

sol.append((M.shape[0],opti.fobj(M, best_pattern),time.time()-start))

for i in sol:
    print(i)

# print("Meilleur pattern trouvé :")
# # print(best_pattern)
# rank, smallest_singular = opti.fobj(M, best_pattern)
# print(f"Rang : {rank}, Plus petite valeur singulière non nulle : {smallest_singular}")
