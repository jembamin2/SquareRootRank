import opti_combi_projet_pythoncode_texte as opti
import numpy as np
import random
from tqdm import tqdm

#%%
def metaheuristic(M, pop_size=100, generations=50, mutation_rate=0.1):
    m, n = M.shape  # Dimensions de la matrice cible
    
    # Génération initiale : population aléatoire avec valeurs {-1, 1}
    def generate_individual():
        return np.random.choice([-1, 1], size=(m, n))  # Matrice avec -1 et 1
    
    def generate_population(size):
        return [generate_individual() for _ in range(size)]

    # Fitness : Évaluation du pattern
    def fitness(individual):
        r, s = opti.fobj(M, individual)
        return r, s  # Rang et plus petite valeur singulière

    # Sélection des parents (tournoi)
    def select_parents(population):
        parent1, parent2 = random.sample(population, 2)
        if opti.compareP1betterthanP2(M, parent1, parent2):
            return parent1
        else:
            return parent2

    # Croisement (crossover)
    def crossover(parent1, parent2):
        mask = np.random.randint(0, 2, size=(m, n))  # Masque binaire pour le croisement
        return np.where(mask, parent1, parent2)

    # Mutation : inverser -1 en 1 et 1 en -1
    def mutate(individual):
        for i in range(m):
            for j in range(n):
                if random.random() < mutation_rate:
                    individual[i, j] *= -1  # Inversion des valeurs
        return individual

    # Initialisation
    population = generate_population(pop_size)
    bestPattern = population[0]
    best_fitness = fitness(bestPattern)

    # Algorithme génétique
    for gen in tqdm(range(generations)):
        # Évaluer la population
        population = sorted(population, key=lambda ind: fitness(ind))
        best_in_gen = population[0]
        best_in_gen_fitness = fitness(best_in_gen)

        # Mettre à jour le meilleur global
        if opti.compareP1betterthanP2(M, best_in_gen, bestPattern):
            bestPattern = best_in_gen
            best_fitness = best_in_gen_fitness

        #print(f"Génération {gen+1} - Meilleur rang : {best_fitness[0]}, Meilleure petite valeur singulière : {best_fitness[1]}")

        # Nouvelle génération
        new_population = [bestPattern]  # Élitisime : garder le meilleur
        while len(new_population) < pop_size:
            parent1 = select_parents(population)
            parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return bestPattern


#%%


#m, n = 6, 4
#M = np.random.rand(m, n)
M = opti.matrices1_ledm(120)

best_pattern = metaheuristic(M, pop_size=70, generations=1000, mutation_rate=0.2)
print("Meilleur pattern trouvé :")
print(best_pattern)

# Évaluation finale
rank, smallest_singular = opti.fobj(M, best_pattern)
print(f"Rang : {rank}, Plus petite valeur singulière non nulle : {smallest_singular}")
