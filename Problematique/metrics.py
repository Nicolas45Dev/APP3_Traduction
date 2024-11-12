# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(x,y):
    # Calcul de la distance d'édition
    matrice_a = np.zeros((len(x)+1, len(y)+1))
    for i in range(len(x)+1):
        for j in range(len(y)+1):
            if i == 0:
                matrice_a[i][j] = j
            elif j == 0:
                matrice_a[i][j] = i
            else:
                if x[i-1] == y[j-1]:
                    matrice_a[i][j] = min(matrice_a[i-1][j] + 1, matrice_a[i][ j-1] + 1, matrice_a[i-1][j-1])
                else:
                    matrice_a[i][j] = min(matrice_a[i-1][j] + 1, matrice_a[i][j-1] + 1, matrice_a[i-1][j-1] + 1)

    return int(matrice_a[len(x)][len(y)])

def confusion_matrix_update(matrix, true, pred):
    for idx1, idx2 in zip(true, pred):
        if 0 <= idx1 < 29 and 0 <= idx2 < 29:
            matrix[idx1, idx2] += 1