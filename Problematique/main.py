# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
from dis import distb
from sys import maxsize

import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import yticks
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

import metrics
from models import *
from dataset import *
from metrics import *
import argparse
import seaborn as sns


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
    training = 0
    test = 1
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    seed = 1
    n_workers = 8
    batch_size = args.n_batches
    train_val_split = .8
    trainval_test_split = .9
    gen_test_images = False
    display_attention = True

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and force_cpu else "cpu")

    dataset = HandwrittenWords('data_trainval.p')
    dataset_test = HandwrittenWords('data_test.p')

    # Séparation du dataset (entraînement et validation)
    n_trainval_samp = int(len(dataset) * trainval_test_split)
    n_test_samp = len(dataset) - n_trainval_samp
    # dataset_trainVal = torch.utils.data.random_split(dataset, [n_trainval_samp])
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])

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
            val_dist = []
            fig, ax = plt.subplots(2, 1)

        # Ignore les symboles de padding
        # criterion = nn.CrossEntropyLoss(ignore_index=dataset.symbol_to_int['<pad>'])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            running_loss_val = 0.0

            model.train()
            dist = 0.0
            dist_val = 0.0

            for batch_idx, (data, target) in enumerate(dataload_train):

                data = data.to(device)

                target = target.to(device)

                optimizer.zero_grad()

                padding_mask = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0) & (data[:, :, 3] == 0)
                padding_mask = ~padding_mask

                output, hidden, attn = model(data, target, padding_mask)

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


            torch.save(model, 'Results/modelSansPad.pt')
            # Validation
            model.eval()

            for data, target in dataload_val:
                data = data.to(device)
                target = target.to(device)

                padding_mask = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0) & (data[:, :, 3] == 0)
                padding_mask = ~padding_mask

                output, hidden, attn = model(data, target, padding_mask)
                loss_val = criterion(output.reshape(-1, dataset.num_character), target.reshape(-1))
                running_loss_val += loss_val.item()

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
                    dist_val += edit_distance(mot_a, mot_b) / M

            print(f'\rEpoch {epoch} - Average Loss: {running_loss / len(dataload_train):.4f} - Average Edit Distance train: {dist / len(dataload_train):.4f} - Average Edit Distance val: {dist_val / len(dataload_val):.4f}')

            if True:
                train_loss.append(running_loss / len(dataload_train))
                val_loss.append(running_loss_val / len(dataload_val))
                train_dist.append(dist / len(dataload_train))
                val_dist.append(dist_val / len(dataload_val))
                ax[0].plot(train_loss)
                ax[0].plot(val_loss)
                ax[0].set_title('Loss')
                ax[0].legend(['Train', 'Validation'])
                ax[1].plot(train_dist)
                ax[1].plot(val_dist)
                ax[1].set_title('Edit Distance')
                ax[1].legend(['Train', 'Validation'])
                plt.draw()

        if True:
            plt.show()
            plt.close('all')

    if test:
        # Évaluation
        model = torch.load('Results/modelSansPad.pt')
        model.eval()
        dist_test = 0.0
        # Charger les données de tests
        # Pour la validation
        # dataset_test = HandwrittenWords('data_test.p')
        dataload_test = DataLoader(dataset_test, batch_size=50, shuffle=False, num_workers=n_workers)
        confusion_matrix = np.zeros([29, 29],dtype=int)
        for i, (data, target) in enumerate(dataload_test):
            data = data.to(device)
            target = target.to(device)

            padding_mask = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0) & (data[:, :, 3] == 0)
            padding_mask = ~padding_mask

            output, hidden, attn = model(data, target, padding_mask)
            output_list = torch.argmax(output, dim=-1).detach().cpu()
            output_one_hot = torch.nn.functional.one_hot(output_list,
                                                      num_classes=dataset.num_character).detach().cpu().numpy()

            # Calcul de la distance d'édition
            target_list = target.detach().cpu().numpy()

            target_labels = target_list.flatten()
            pred_labels = output_list.flatten()
            metrics.confusion_matrix_update(confusion_matrix, target_labels, pred_labels)
            M = len(output_list)
            for i in range(M):
                a = output_one_hot[i]
                b = target_list[i]
                mot_a = "".join(dataset.onehot_to_string(a))
                mot_b = "".join(dataset.int_to_string(b))
                dist_test += edit_distance(mot_a, mot_b) / M
        print(f'Average Edit Distance: {dist_test / len(dataload_test):.4f}')

        # Affichage des résultats de test
        for i in range(10):
            output, target = dataset_test[np.random.randint(0, len(dataset_test))]
            data = output.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            padding_mask = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0) & (data[:, :, 3] == 0)
            padding_mask = ~padding_mask

            output, hidden, attn = model(data, target, padding_mask)
            output_list = torch.argmax(output, dim=-1).detach().cpu()
            output_one_hot = torch.nn.functional.one_hot(output_list, num_classes=dataset.num_character).detach().cpu().numpy()

            target_list = target.detach().cpu().numpy()
            mot_a = "".join(dataset.onehot_to_string(output_one_hot[0], pad=False))
            mot_b = "".join(dataset.int_to_string(target_list[0], pad=False))
            print('Output: ', mot_a)
            print('Target: ', mot_b)
            print("")

            if display_attention:
                data_no_device = data.detach().cpu()
                x_points = data_no_device[0, :, 0] # data[0, :, 1].detach().cpu().numpy()  # Points x (1ère séquence de batch)
                y_points = data_no_device[0, :, 1]  # Points y

                # Taille de la séquence (nombre de points)
                seq_len = x_points.shape[0]

                # Extraire les poids d'attention (de taille 7, 457)
                attn = attn.detach().cpu().numpy()[0, :, :]  # Prendre la première séquence du batch

                # Créer une figure
                plt.figure()

                # Afficher les points manuscrits
                plt.scatter(x_points, y_points, c='blue', label='Tracé manuscrit')

                # Affichage des poids d'attention sous forme de heatmap
                # Normaliser la taille de la heatmap pour correspondre à la taille de l'image des points
                attention_map = np.zeros((seq_len, seq_len))
                for i in range(attn.shape[0]):  # Pour chaque timestep dans le décodeur
                    attention_map[i, :attn.shape[1]] = attn[i]

                # Afficher les poids d'attention comme heatmap sur les points
                # plt.imshow(attention_map.T, cmap='hot', origin='lower', alpha=0.5, extent=[0, 1, 0, 1])

                # plt.colorbar(label='Poids d\'attention')
                plt.xlabel('Position x')
                plt.ylabel('Position y')
                plt.title('Poids d\'Attention sur le Tracé Manuscrit')
                # plt.legend()
                plt.show()



        
        # Affichage de la matrice de confusion
        # conf_matrix = np.sum(confusion_matrix_array, dtype=np.int64, axis=0)
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=dataset.int_to_string(np.arange(29), pad=False), yticklabels=dataset.int_to_string(np.arange(29), pad=False))
        plt.xlabel('Prediction')
        plt.ylabel('Target')
        plt.title('Confusion Matrix')
        plt.show()