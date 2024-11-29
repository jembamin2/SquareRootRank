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

import numpy as np
import random

def metaheuristic(M, 
                  pop_size, 
                  generations, 
                  mutation_rate, 
                  num_parents, 
                  num_children):
    start=time.time()
    
    m, n = M.shape  # Dimensions de la matrice cible
    
    # Génération initiale : population aléatoire avec valeurs {-1, 1}
    def generate_individual():
        return np.random.choice([-1, 1], size=(m, n))
    
    def generate_population(size):
        return [generate_individual() for _ in range(size)]

    # Fitness : Évaluation du pattern
    def fitness(individual):
        r, s = opti.fobj(M, individual)
        return r, s

    # Méthodes de sélection des parents
    def select_parents_tournament(population, num_parents):
        """Sélection par tournoi."""
        selected_parents = []
        for _ in range(num_parents):
            competitors = random.sample(population, 3)
            competitors = sorted(competitors, key=lambda ind: fitness(ind))
            selected_parents.append(competitors[0])  # Le meilleur gagne
        return selected_parents
    
    def select_parents_roulette(population, num_parents):
        """Sélection par roulette basée sur l'inverse du rang."""
        fitness_values = [1 / (fitness(ind)[0] + 1e-6) for ind in population]
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        return random.choices(population, probabilities, k=num_parents)

    # Liste des méthodes de sélection des parents
    parent_selection_methods = [
        select_parents_tournament,
        select_parents_roulette
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
    def mutate(individual):
        for i in range(m):
            for j in range(n):
                if random.random() < mutation_rate:
                    individual[i, j] *= -1
        return individual

    # Méthode pour sélectionner la nouvelle génération
    def generation_elitism(parents, children, population):
            combined = parents + children
            return sorted(combined, key=lambda ind: fitness(ind))[:pop_size]
    
    def generation_combined(parents, children, population):
            return sorted(population + children, key=lambda ind: fitness(ind))[:pop_size]
    
    new_generation_methods = [
        generation_elitism, 
        generation_combined
    ]

    # Initialisation
    population = generate_population(pop_size)
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
            
            # Mutation
            child = mutate(child)
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
        if best_fitness[0] == 2 or time.time()-start>300:
            break

    
    return bestPattern



#%%

# M = read_matrix("test(pas unitaire)/correl5_matrice.txt")


#m, n = 6, 4
#M = np.random.rand(m, n)
M = opti.matrices1_ledm(100)

best_pattern = metaheuristic(
    M, 
    pop_size=80,
    generations=1000000, 
    mutation_rate=0.2, 
    num_parents=80, 
    num_children=300
)



print("Meilleur pattern trouvé :")
print(best_pattern)
rank, smallest_singular = opti.fobj(M, best_pattern)
print(f"Rang : {rank}, Plus petite valeur singulière non nulle : {smallest_singular}")