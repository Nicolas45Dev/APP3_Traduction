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
    parser.add_argument('--n_batches', type=int, help='Nombre de batchs')
    parser.add_argument('--learning_rate', type=float, help='Affichage des courbes d\'entrainement')
    parser.add_argument('--n_epochs', type=int, default=50, help='Nombre d\'époques')

    args = parser.parse_args()

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = args.force_cpu
    training = args.training
    test = args.test
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    seed = 1
    n_workers = 4
    batch_size = args.n_batches
    train_val_split = .8
    trainval_test_split = .9
    gen_test_images = False
    display_attention = False

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and force_cpu else "cpu")

    dataset = HandwrittenWords('data_trainval.p')
    # Séparation du dataset (entraînement et validation)
    n_trainval_samp = int(len(dataset) * trainval_test_split)
    n_test_samp = len(dataset) - n_trainval_samp
    dataset_trainVal, dataset_test = torch.utils.data.random_split(dataset, [n_trainval_samp, n_test_samp])
    n_train_samp = int(len(dataset_trainVal) * train_val_split)
    n_val_samp = len(dataset_trainVal) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_trainVal, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset_train))
    print('Validation data : ', len(dataset_val))
    print('\n')
    model = trajectory2seq(n_hidden=10, n_layers=2, dict_size=dataset.dict_size, device=device, max_len=MAX_LEN + 1, symbol_to_int=dataset.symbol_to_int)

    print("Nombre d'époques:", n_epochs)
    print("Ensemble de données:", len(dataset))
    print("Modèle:", model)
    print("Nombre de paramètres:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Device:", device)
    print("\n")

    if training:

        # Affichage
        if True:
            train_dist = []
            train_loss = []
            val_loss = []
            fig, ax = plt.subplots(2, 1)

        # Ignore les symboles de padding
        criterion = nn.CrossEntropyLoss()#ignore_index=dataset.symbol_to_int['<pad>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            running_loss_val = 0.0

            model.train()
            dist = 0.0

            for batch_idx, (data, target) in enumerate(dataload_train):

                data = data.to(device)

                target = target.to(device)

                optimizer.zero_grad()
                output, hidden, attn = model(data, target)

                # Calcul de la loss
                # target_cross_entropy = torch.argmax(target, dim=-1).long()
                loss = criterion(output.reshape(-1, dataset.num_character), target.reshape(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu()
                # output_list = dataset.int_to_onehot(output_list)
                output_list = torch.nn.functional.one_hot(output_list, num_classes=dataset.num_character).detach().cpu().numpy()
                target_list = target.detach().cpu().numpy()
                M = len(output_list)
                for i in range(M):
                    a = output_list[i]
                    b = target_list[i]
                    mot_a = dataset.onehot_to_string(a)
                    mot_b = dataset.int_to_string(b)
                    dist += edit_distance(mot_a, mot_b) / M


            torch.save(model, 'model.pt')
            # Validation
            model.eval()

            for data, target in dataload_val:
                data = data.to(device)
                target = target.to(device)

                output, hidden, attn = model(data, target)
                loss_val = criterion(output.reshape(-1, dataset.num_character), target.reshape(-1))
                running_loss_val += loss_val.item()

            print(f'\rEpoch {epoch} - Average Loss: {running_loss / len(dataload_train):.4f} - Average Edit Distance: {dist / len(dataload_train):.4f}')

            if True:
                train_loss.append(running_loss / len(dataload_train))
                val_loss.append(running_loss_val / len(dataload_val))
                train_dist.append(dist / len(dataload_train))
                ax[0].plot(train_loss)
                ax[0].plot(val_loss)
                ax[0].set_title('Loss')
                ax[0].legend(['Train', 'Validation'])
                ax[1].plot(train_dist)
                ax[1].set_title('Edit Distance')
                plt.draw()

        if True:
            plt.show()
            plt.close('all')

    if test:
        # Évaluation
        # À compléter
        model = torch.load('model.pt')
        model.eval()
        # Charger les données de tests
        # À compléter
        # Pour la validation
        # dataset_test = HandwrittenWords('data_test.p')
        dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        for data, target in dataload_test:
            data = data.to(device)
            target = target.to(device)
            output, hidden, attn = model(data, target)
            output_list = torch.argmax(output, dim=-1).detach().cpu()
            output_list = torch.nn.functional.one_hot(output_list,
                                                      num_classes=dataset.num_character).detach().cpu().numpy()
            target_list = target.detach().cpu().numpy()
        # if display_attention:
        #     attn = attn.detach().cpu().numpy()
        #     plt.figure()
        #     plt.imshow(attn[0])
        #     plt.xticks(np.arange(0, 32, 1))
        #     plt.yticks(np.arange(0, 32, 1))
        #     plt.show()

        # Affichage des résultats de test
            for i in range(len(output_list)):
                a = output_list[i]
                b = target_list[i]
                mot_a = dataset.onehot_to_string(a, pad=False)
                mot_b = dataset.int_to_string(b, pad=False)
                print('Output: ', mot_a)
                print('Target: ', mot_b)
                print('')
        
        # Affichage de la matrice de confusion
        # À compléter

        pass