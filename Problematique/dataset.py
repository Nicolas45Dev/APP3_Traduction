import numpy
import torch
import numpy as np
from fontTools.ttx import ttList
from sympy.physics.units import speed
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle

MAX_LEN = 5 + 1

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol = '<pad>'
        self.start_symbol = '<sos>'
        self.stop_symbol = '<eos>'
        self.symbol_to_onehot = {
            'a': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'd': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'g': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'i': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'j': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'l': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'n': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'o': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'p': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            's': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            't': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'u': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'v': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'w': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'x': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            self.start_symbol: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            self.stop_symbol: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            self.pad_symbol: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        self.int_to_onehot = {
            0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            12: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            14: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            17: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            19: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            20: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            21: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            22: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            23: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            24: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            25: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            26: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            27: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            28: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        self.symbol_to_int = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7,
            'i': 8,
            'j': 9,
            'k': 10,
            'l': 11,
            'm': 12,
            'n': 13,
            'o': 14,
            'p': 15,
            'q': 16,
            'r': 17,
            's': 18,
            't': 19,
            'u': 20,
            'v': 21,
            'w': 22,
            'x': 23,
            'y': 24,
            'z': 25,
            self.start_symbol: 26,
            self.stop_symbol: 27,
            self.pad_symbol: 28
        }

        self.num_character = 29

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        self.data = dict(self.data)
        self.keys = list(self.data.keys())
        self.keys = [list(word) for word in self.keys]
        self.values = list(self.data.values())
        self.dict_size = len(self.data.keys())

        self.max_sequence_len = max([len(point[1]) for point in self.values])

        # Soustraire le premier point de chaque séquence
        self.first_point_x = [point[0][0] for point in self.values]
        self.first_point_y = [point[1][0] for point in self.values]

        for i in range(len(self.values)):
            self.values[i][0] = [self.values[i][0][j] - self.first_point_x[i] for j in range(len(self.values[i][0]))]
            self.values[i][1] = [self.values[i][1][j] - self.first_point_y[i] for j in range(len(self.values[i][1]))]

        # self.norm_data()
        # self.standardize_data()
        self.difference_point()
        # self.second_derivative()

        # Ajout du padding aux séquences
        self.keys = [[self.start_symbol] + word + [self.stop_symbol] + [self.pad_symbol] * (MAX_LEN - len(word) - 1) for word in self.keys]

        padded_values = []

        for sequence in self.values:
            # Taille actuelle de la séquence
            current_len = sequence.shape[1]

            # Calculer le padding nécessaire
            if current_len < self.max_sequence_len:
                # Pad avec des zéros pour atteindre la longueur maximale
                padding_size = self.max_sequence_len - current_len
                # Créer le padding avec numpy de taille (2, padding_size)
                padding = np.zeros((2, padding_size))
                # Concaténer la séquence originale avec le padding
                padded_sequence = np.concatenate((sequence, padding), axis=1)
            else:
                # Si la séquence est déjà de la taille max, on la tronque
                padded_sequence = sequence[:, :self.max_sequence_len]

            padded_values.append(padded_sequence)

        # Convertir en tensor PyTorch de taille (4880, 2, self.max_sequence_len)
        padded_values_tensor = torch.tensor(np.array(padded_values), dtype=torch.float32)

        padded_speed = []

        for sequence in self.speed:
            # Taille actuelle de la séquence
            current_len = sequence.shape[1]

            # Calculer le padding nécessaire
            if current_len < self.max_sequence_len:
                # Pad avec des zéros pour atteindre la longueur maximale
                padding_size = self.max_sequence_len - current_len
                # Créer le padding avec numpy de taille (2, padding_size)
                padding = np.zeros((2, padding_size))
                # Concaténer la séquence originale avec le padding
                padded_sequence = np.concatenate((sequence, padding), axis=1)
            else:
                # Si la séquence est déjà de la taille max, on la tronque
                padded_sequence = sequence[:, :self.max_sequence_len]

            padded_speed.append(padded_sequence)

        # Convertir en tensor PyTorch de taille (4880, 2, self.max_sequence_len)
        padded_speed_tensor = torch.tensor(np.array(padded_speed), dtype=torch.float32)

        self.values = torch.stack([padded_values_tensor, padded_speed_tensor], dim=1)

        self.data = list(zip(self.keys, self.values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        key = [self.symbol_to_int[i] for i in key]

        # Aplatir la liste des valeurs
        values_tensor = torch.as_tensor(self.values[idx], dtype=torch.float64).clone().detach()
        values_tensor = torch.stack((values_tensor[0][0], values_tensor[0][1], values_tensor[1][0], values_tensor[1][1])).T
        key_tensor = torch.as_tensor(key, dtype=torch.long)

        return values_tensor, key_tensor


    def visualisation(self, idx):
        # Plot the handwritten word
        fig = plt.figure()
        plt.title(self.keys[idx])
        plt.scatter(self.data[idx][1][0][0], self.data[idx][1][0][1])
        # plt.scatter(self.data[idx][1][1][0], self.data[idx][1][1][1])
        plt.show()

    def onehot_to_string(self, onehot, pad=True):
        # Convertir une liste de onehot en string
        mot = [list(self.symbol_to_onehot.keys())[i] for i in np.argmax(onehot, axis=1)]

        # Enlever les symboles de start, stop et padding
        if pad:
            mot = [i for i in mot if i not in [self.start_symbol, self.stop_symbol, self.pad_symbol]]

        return mot

    def int_to_string(self, int_list, pad=True):
        # Convertir une liste d'entiers en string
        mot = [list(self.symbol_to_int.keys())[i] for i in int_list]

        # Enlever les symboles de start, stop et padding
        if pad:
            mot = [i for i in mot if i not in [self.start_symbol, self.stop_symbol, self.pad_symbol]]

        return mot

    def standardize_data(self):
        """
        La fonction va standardisé les données
        """
        avg_x = ([np.mean(word[0]) for word in self.values])
        avg_y = ([np.mean(word[1]) for word in self.values])
        std_x = ([np.std(word[0]) for word in self.values])
        std_y = ([np.std(word[1]) for word in self.values])

        for i in range(len(self.values)):
            self.values[i][0] = (self.values[i][0] - avg_x[i]) / std_x[i]
            self.values[i][1] = (self.values[i][1] - avg_y[i]) / std_y[i]

    def norm_data(self):
        """
        La fonction va normaliser les données
        """
        max_x = ([np.max(word[0]) for word in self.values])
        max_y = ([np.max(word[1]) for word in self.values])
        min_x = ([np.min(word[0]) for word in self.values])
        min_y = ([np.min(word[1]) for word in self.values])

        for i in range(len(self.values)):
            self.values[i][0] = (self.values[i][0] - min_x[i]) / (max_x[i] - min_x[i])
            self.values[i][1] = (self.values[i][1] - min_y[i]) / (max_y[i] - min_y[i])

    def difference_point(self):
        """
        La fonction va calculer la différence entre les points
        """

        # self.speed un array comme self.values
        self.speed = [ np.array([np.zeros(len(word[0])), np.zeros(len(word[1]))]) for word in self.values]

        for i in range(len(self.values)):
            self.speed[i][0] = np.append(np.diff(self.values[i][0]), np.zeros(1))
            self.speed[i][1] = np.append(np.diff(self.values[i][1]), np.zeros(1))



if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_test.p')
    item = a.__getitem__(1)

    for i in range(4):
        a.visualisation(np.random.randint(0, len(a)))