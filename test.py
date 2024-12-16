import numpy as np

# Dimensions de la matrice
m, n, r = 120, 120, 10

# Créer une ligne de base aléatoire
base_line = ((np.random.rand(m, r) * 10)@(np.random.rand(r, n) * 10)) ** 2

# Taille de la matrice
rows, cols = base_line.shape
size_info = f"{rows} {cols}\n"

# Sauvegarder la matrice dans un fichier
with open('file.txt', 'w') as f:
    f.write(size_info)
    np.savetxt(f, base_line, fmt='%d')

print("La taille de la matrice et les données ont été sauvegardées dans file.txt")
