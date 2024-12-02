import opti_combi_projet_pythoncode_texte as opti
import numpy as np
import random
from tqdm import tqdm
import time

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
                  num_children):
    start=time.time()
    
    m, n = M.shape  # Dimensions de la matrice cible
    indices = [(i, j) for i in range(m) for j in range(n)]
    
    # Génération initiale : population aléatoire avec valeurs {-1, 1}
    # def generate_individual():
    #     return np.random.choice([-1, 1], size=(m, n))
    
    # def generate_population(size):
    #     return [generate_individual() for _ in range(size)]
    
    def generate_clever_individual():
        individual = np.ones((m, n))
        for i in range(m):
            for j in range(n):
                if M[i, j] < np.median(M):
                    individual[i, j] = -1
        return individual
    
    def generate_population(size, sa_solutions):
        population = [generate_clever_individual() for _ in range(size-len(sa_solutions))]
        return sa_solutions + population
    
    # def average_singular_value(population):
    #     singular_values = [opti.fobj(M, ind)[1] for ind in population]
    #     return np.mean(singular_values)
    
    # # Print average singular value for clever individuals
    # clever_population = [generate_clever_individual() for _ in range(pop_size)]
    # avg_singular_value_clever = average_singular_value(clever_population)
    # print(f"Average singular value (clever individuals): {avg_singular_value_clever}")

    # # Print average singular value for random individuals
    # random_population = [generate_individual() for _ in range(pop_size)]
    # avg_singular_value_random = average_singular_value(random_population)
    # print(f"Average singular value (random individuals): {avg_singular_value_random}")
    

    # Fitness : Évaluation du pattern
    def fitness(individual):
        r, s = opti.fobj(M, individual)
        return r, s

    # Méthodes de sélection des parents
    import random

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
        swap_indices = random.sample(indices, 1)
        for i, j in swap_indices:
            individual[i, j] *= -1
        return individual

    def mutate_5(individual):
        swap_indices = random.sample(indices, 5)
        for i, j in swap_indices:
            individual[i, j] *= -1
        return individual

    def mutate_swap_all(individual):
        return -individual

    mutate_methods = [
        mutate_many,
        mutate_1,
        mutate_5,
        mutate_swap_all
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
    population = (sorted(generate_population(pop_size, sa_solutions), key=lambda ind: fitness(ind)))
    bestPattern = population[0]
    best_fitness = fitness(bestPattern)

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
                    weights=[0.25, 0.4, 0.4, 0.25], 
                    k=1
                )[0]
                child = mutate_method(child)
            children.append(child)
        
        # Nouvelle génération
        new_generation_method = random.choice(new_generation_methods)
        population = new_generation_method(parents, children, population)

        # Mettre à jour le meilleur pattern
        best_in_pop = sorted(population, key=lambda ind: fitness(ind))[0]
        if opti.compareP1betterthanP2(M, best_in_pop, bestPattern):
            bestPattern = best_in_pop
            best_fitness = fitness(bestPattern)
            # print(best_fitness)

        print(f"Génération {gen+1} - Rang : {best_fitness[0]}, Petite valeur singulière : {best_fitness[1]}")
        # if best_fitness[0] == 2:
        #     break

        if time.time()-start>300:
            break

    
    return bestPattern



#%%

# M = read_matrix("test(pas unitaire)/correl5_matrice.txt")
# M = read_matrix("test(pas unitaire)/slack7gon_matrice.txt")
# M = read_matrix("test(pas unitaire)/synthetic_matrice.txt")

# m, n = 10, 10
# M = np.random.rand(m, n)
M = opti.matrices1_ledm(14)

sol = []
sa_solutions=[]

for i in range(2):
    best_pattern = metaheuristic(
        M, 
        sa_solutions,
        pop_size=250,               # Population size (includes SA solutions)
        generations=500, 
        mutation_rate=0.35, 
        num_parents=80, 
        num_children=200
    )
    sol.append(opti.fobj(M, best_pattern))

for i in sol:
    print(i)

# print("Meilleur pattern trouvé :")
# # print(best_pattern)
# rank, smallest_singular = opti.fobj(M, best_pattern)
# print(f"Rang : {rank}, Plus petite valeur singulière non nulle : {smallest_singular}")
