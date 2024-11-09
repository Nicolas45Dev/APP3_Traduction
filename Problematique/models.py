# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, dict_size, device, max_len):
        super(trajectory2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2 * n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)
        self.encoder2value = nn.Linear(self.dict_size, n_hidden)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size)
        self.to(device)

    def encoder(self, x):
        # Encodeur
        out, hidden = self.encoder_layer(self.fr_embedding(x))
        out = self.fc(out)
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
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(
            self.device)  # Vecteur de sortie du décodage
        attention_weights = torch.zeros((batch_size, self.max_len, self.max_len[)).to(
            self.device)  # Poids d'attention
        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            out, hidden = self.decoder_layer(self.en_embedding(vec_in), hidden)
            attention_out, weights = self.attentionModule(out, encoder_outs)
            vec_in = self.att_combine(torch.cat((out, attention_out), dim=2))
            vec_in = self.fc(vec_in)
            vec_out[:, i] = vec_in[:, 0]
            vec_in = torch.argmax(out, dim=2)
            attention_weights[:, :, i] = weights[:, 0]
        return vec_out, hidden, attention_weights

    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out, h)
        return out, hidden, attn

