import numpy as np

# Créer une ligne de base aléatoire
base_line = np.random.randint(0, 50, size=50)

# Générer une matrice où chaque ligne est une variation légère de base_line
randMat = np.array([base_line + np.random.randint(0, 5, size=50) for _ in range(50)])

# Sauvegarder la matrice dans un fichier
np.savetxt('file.txt', randMat, fmt='%d')

print("randMat has been saved to file.txt")
