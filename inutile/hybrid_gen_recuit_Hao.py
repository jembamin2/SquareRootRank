import opti_combi_projet_pythoncode_texte_v2 as opti
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
    def select_parents_tournament(population, num_parents):
        """Sélection par tournoi."""
        selected_parents = []
        for _ in range(num_parents):
            competitors = random.sample(population, 2)
            competitors = sorted(competitors, key=lambda ind: fitness(ind))
            selected_parents.append(competitors[0])  # Le meilleur gagne
        return selected_parents
    
    def select_parents_roulette(population, num_parents):
        """Sélection par roulette."""
        fitness_values = [1/(fitness(ind)[1]) for ind in population]
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
    # print(f"Population initiale triée : {population}")
    bestPattern = population[0]
    best_fitness = fitness(bestPattern)
    

    # Algorithme génétique
    for gen in tqdm(range(generations)):
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
                    weights=[0.2, 0.4, 0.4, 0.2], 
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

        # print(f"Génération {gen+1} - Rang : {best_fitness[0]}, Petite valeur singulière : {best_fitness[1]}")
        # if best_fitness[0] == 2:
        #     break

        if time.time()-start>300:
            break

    
    return bestPattern

def simulated_annealing(M, 
                        initial_temp, 
                        final_temp, 
                        alpha, 
                        max_iter_per_temp, 
                        mutation_rate):
    
    m, n = M.shape

    # Générer une solution initiale aléatoire
    def generate_solution():
        return np.random.choice([-1, 1], size=(m, n))

    # Fonction de voisinage : mutation d'une fraction aléatoire des éléments
    def generate_neighbor(solution):
        neighbor = solution.copy()
        for i in range(m):
            for j in range(n):
                if random.random() < mutation_rate:
                    neighbor[i, j] *= -1  # Inversion de -1 à 1 et vice versa
        return neighbor

    # Fonction d'évaluation
    def fitness(solution):
        rank, singular_value = opti.fobj(M, solution)
        return rank, singular_value

    # Probabilité d'acceptation d'une solution moins bonne
    def acceptance_probability(current_fitness, neighbor_fitness, temp):
        delta_fitness = neighbor_fitness[0] - current_fitness[0]
        # Si le rang est identique, considérer la plus petite valeur singulière
        if delta_fitness == 0:
            delta_fitness = neighbor_fitness[1] - current_fitness[1]
        if delta_fitness < 0:  # Meilleure solution
            return 1.0
        return np.exp(-delta_fitness / temp)

    # Initialisation
    current_solution = generate_solution()
    current_fitness = fitness(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness

    temperature = initial_temp
    start_time = time.time()

    # Boucle principale
    while temperature > final_temp:
        for _ in range(max_iter_per_temp):
            # Générer un voisin
            neighbor = generate_neighbor(current_solution)
            neighbor_fitness = fitness(neighbor)

            # Décider si on accepte le voisin
            if random.random() < acceptance_probability(current_fitness, neighbor_fitness, temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                # Mettre à jour la meilleure solution si nécessaire
                if opti.compareP1betterthanP2(M, current_solution, best_solution):
                    best_solution = current_solution
                    best_fitness = current_fitness
                    print(f"Amélioration : Rang {best_fitness[0]}, SVD {best_fitness[1]}")
            
            # Arrêter si la solution optimale est trouvée
            # if best_fitness[0] == 2:
            #     return best_solution, best_fitness
            if time.time() - start_time > 300:
                return best_solution, best_fitness

        # Refroidir la température
        temperature *= alpha
        print(f"Température : {temperature:.4f}, Rang actuel : {current_fitness[0]}, Meilleur rang : {best_fitness[0]}")

    return best_solution, best_fitness


#%%

# M = read_matrix("test(pas unitaire)/correl5_matrice.txt")
# M = read_matrix("test(pas unitaire)/slack7gon_matrice.txt")
# M = read_matrix("test(pas unitaire)/synthetic_matrice.txt")

#m, n = 6, 4
#M = np.random.rand(m, n)
M = opti.matrices1_ledm(20)

# Run Simulated Annealing 10 times and collect solutions
sa_solutions = []
for i in range(10):
    print(f"Simulated Annealing Run {i+1}/10")
    sa_solution, sa_fitness = simulated_annealing(
        M, 
        initial_temp=100,
        final_temp=5e-4,
        alpha=0.95,
        max_iter_per_temp=1000,
        mutation_rate=0.2
    )
    sa_solutions.append(sa_solution)

# Use SA solutions as the initial population in the genetic algorithm
best_pattern = metaheuristic(
    M, 
    sa_solutions,
    pop_size=100,               # Population size (includes SA solutions)
    generations=1000, 
    mutation_rate=0.35, 
    num_parents=30, 
    num_children=100
)



print("Meilleur pattern trouvé :")
# print(best_pattern)
rank, smallest_singular = opti.fobj(M, best_pattern)
print(f"Rang : {rank}, Plus petite valeur singulière non nulle : {smallest_singular}")
