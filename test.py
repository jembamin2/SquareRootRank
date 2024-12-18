import numpy as np

# Dimensions de la matrice
m, n, r =30, 30, 4

# Generate random matrices with values between 0 and 1
matrix_mr = np.random.rand(m, r)
matrix_rn = np.random.rand(r, n)

# Generate random masks with values of -1 or 1
mask_mr = np.random.choice([-1, 1], size=(m, r))
mask_rn = np.random.choice([-1, 1], size=(r, n))

# Apply the masks
masked_mr = matrix_mr * mask_mr
masked_rn = matrix_rn * mask_rn

# Perform the matrix multiplication and element-wise power
base_line = (masked_mr @ masked_rn) ** 2

# Taille de la matrice
rows, cols = base_line.shape
size_info = f"{rows} {cols}\n"

# Sauvegarder la matrice dans un fichier
with open('file.txt', 'w') as f:
    f.write(size_info)
    np.savetxt(f, base_line)

print("La taille de la matrice et les données ont été sauvegardées dans file.txt")
