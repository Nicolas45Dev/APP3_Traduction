# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, dict_size, device, max_len, max_point_len = 914):
        super(trajectory2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.dict_size = dict_size
        self.max_len = max_len
        self.point_size = max_point_len

        # Définition des couches du rnn
        self.encoder_layer = nn.GRU(2, n_hidden, n_layers, batch_first=True, dtype=torch.float64)
        self.decoder_layer = nn.GRU(4, n_hidden, n_layers, batch_first=True, dtype=torch.float64)
        self.embedding = nn.Embedding(dict_size, n_hidden, dtype=torch.float64)

        # Définition de la couche dense pour l'attention
        # self.att_combine = nn.Linear(2 * n_hidden, n_hidden)
        # self.hidden2query = nn.Linear(n_hidden, n_hidden)
        # self.encoder2value = nn.Linear(self.dict_size, n_hidden)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(4, 4880, dtype=torch.float64)
        self.fc1 = nn.Linear(4, 4880, dtype=torch.float64)
        self.to(device)

    def encoder(self, x):
        # Encodeur
        out, hidden = self.encoder_layer(x)
        out = self.fc(hidden[-1])
        return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)
        values = self.encoder2value(values)

        # Attention
        similarity = torch.bmm(query, values.transpose(1, 2))
        attention_weights = torch.nn.functional.softmax(similarity, dim=2)
        attention_output = torch.bmm(attention_weights, values)

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        # Décodeur avec attention
        # Initialisation des variables
        max_len = self.max_len # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch

        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, 6, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage

        # attention_weights = torch.zeros((batch_size, self.max_len, self.max_len)).to(self.device)

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            out, hidden = self.decoder_layer(self.embedding(vec_in), hidden)
            out = self.fc1(out)
            vec_in = torch.argmax(out, dim=2)
            vec_out[:, i] = out[:, 0]

        return vec_out, hidden, None

    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out, h)
        return out, h, None

