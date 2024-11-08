# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np
from sklearn.metrics import confusion_matrix as cm

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

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion
    if len(ignore) > 0:
        confusion = cm(true, pred, labels=ignore)
    else:
        confusion = cm(true, pred)

    return confusion
