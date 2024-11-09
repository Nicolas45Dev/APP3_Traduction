# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
from dis import distb

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *
import argparse

if __name__ == '__main__':

    # Obtenir les hyperparamètres à partir des paramètres de programme
    parser = argparse.ArgumentParser(description='GRO722 - Problématique')
    parser.add_argument('--force_cpu', type=bool, help='Forcer a utiliser le cpu')
    parser.add_argument('--training', type=bool, help='Entrainement')
    parser.add_argument('--test', type=bool, help='Test')
    parser.add_argument('--learning_curves', type=float, help='Affichage des courbes d\'entrainement')
    parser.add_argument('--n_epochs', type=int, default=50, help='Nombre d\'époques')

    args = parser.parse_args()

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = args.force_cpu
    training = args.training
    test = args.test
    learning_curves = args.learning_curves
    n_epochs = args.n_epochs
    seed = 1
    n_workers = 4
    gen_test_images = False

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    dataset = HandwrittenWords('data_trainval.p')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=n_workers)

    model = trajectory2seq(n_hidden=4, n_layers=2, dict_size=dataset.dict_size, device=device, max_len=MAX_LEN)

    print("Nombre d'époques:", n_epochs)
    print("Ensemble de données:", len(dataset))
    print("Modèle:", model)
    print("Nombre de paramètres:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Device:", device)
    print("\n")

    if training:

        # Ignore les symboles de padding
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_symbol)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            dist = 0.0

            for batch_idx, (data, target) in enumerate(dataloader):

                optimizer.zero_grad()
                output, hidden, attn = model(data)
            
            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass