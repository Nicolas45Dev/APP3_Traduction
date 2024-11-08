import numpy as np
import time
    
def edit_distance(a,b):
    # Calcul de la distance d'édition

    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------

    matrice_a = np.zeros((len(a)+1, len(b)+1))



    for i in range(len(a)+1):
        for j in range(len(b)+1):
            if i == 0:
                matrice_a[i][j] = j
            elif j == 0:
                matrice_a[i][j] = i
            else:
                if a[i-1] == b[j-1]:
                    matrice_a[i][j] = min(matrice_a[i-1][j] + 1, matrice_a[i][ j-1] + 1, matrice_a[i-1][j-1])
                else:
                    matrice_a[i][j] = min(matrice_a[i-1][j] + 1, matrice_a[i][j-1] + 1, matrice_a[i-1][j-1] + 1)

    return int(matrice_a[len(a)][len(b)])
    
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------

if __name__ =="__main__":
    a = list('chien')
    b = list('chat')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
