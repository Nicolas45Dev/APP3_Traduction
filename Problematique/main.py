# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
from dis import distb

import matplotlib.pyplot as plt
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
    device = torch.device("cuda" if torch.cuda.is_available() and force_cpu else "cpu")

    dataset = HandwrittenWords('data_trainval.p')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=n_workers)

    model = trajectory2seq(n_hidden=12, n_layers=2, dict_size=dataset.dict_size, device=device, max_len=MAX_LEN)

    print("Nombre d'époques:", n_epochs)
    print("Ensemble de données:", len(dataset))
    print("Modèle:", model)
    print("Nombre de paramètres:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Device:", device)
    print("\n")

    if training:

        # Affichage
        if learning_curves:
            train_dist = []
            train_loss = []
            fig, ax = plt.subplots(2, 1)

        # Ignore les symboles de padding
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            model.train()
            dist = 0.0

            for batch_idx, (data, target) in enumerate(dataloader):

                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output, hidden, attn = model(data)

                # Calcul de la loss
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu()
                output_list = torch.nn.functional.one_hot(output_list, num_classes=dataset.num_character).detach().cpu().numpy()
                target_list = target.detach().cpu().numpy()
                M = len(output_list)
                for i in range(M):
                    a = output_list[i]
                    b = target_list[i]
                    mot_a = dataset.onehot_to_string(a)
                    mot_b = dataset.onehot_to_string(b)
                    dist += edit_distance(mot_a, mot_b) / M

                # print(f'\rEpoch {epoch} - Batch {batch_idx}/{len(dataloader)} - Average Loss: {loss.item():.4f}', end='')

            print(f'\rEpoch {epoch} - Average Loss: {running_loss/len(dataloader):.4f} - Average Edit Distance: {dist / len(dataloader):.4f}')

            if learning_curves:
                train_loss.append(running_loss/len(dataloader))
                train_dist.append(dist/len(dataloader))
                ax[0].plot(train_loss)
                ax[0].set_title('Loss')
                ax[1].plot(train_dist)
                ax[1].set_title('Edit Distance')
                plt.draw()

        if learning_curves:
            plt.show()
            plt.close('all')

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