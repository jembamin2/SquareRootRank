import numpy as np

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

from scipy.linalg import circulant
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

def metaheuristic(M):
  bestPattern = np.ones(M.shape) #pattern initial

  ... #votre méthode

  return bestPattern

# M = np.array([[4,0,1],[1,1,1],[1,1,0]])
# P1 = np.array([[1,1,-1],[-1,1,1],[1,-1,-1]])
# P2 = np.array([[-1,1,-1],[-1,-1,1],[1,1,-1]])
# print(compareP1betterthanP2(M,P1,P2))
# print(np.linalg.svd(P1*np.sqrt(M), compute_uv=False))

# M = matrices2_slackngon(7)
# P = np.array([[1,1,1,1,1,-1,1],[1,1,1,-1,1,-1,1],[1,1,1,1,1,1,-1],[1,-1,1,1,1,-1,-1],[1,1,-1,1,1,1,1],[1,-1,1,-1,-1,1,1],[1,1,1,1,1,1,1]])
# print(fobj(M,P))