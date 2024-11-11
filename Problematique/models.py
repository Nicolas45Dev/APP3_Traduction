# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import random

import torch
from numpy.matlib import randn
from torch import nn
from torch.nn.functional import embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt


class trajectory2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, dict_size, device, max_len, symbol_to_int):
        super(trajectory2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.dict_size = dict_size
        self.max_len = max_len
        self.symbol_to_int = symbol_to_int

        self.pad_symbol = '<pad>'
        self.start_symbol = '<sos>'
        self.stop_symbol = '<eos>'
        self.device = device
        self.teaching_forcing_ratio = 0.6
        # Définition des couches du rnn
        self.encoder_layer = nn.LSTM(2, n_hidden, n_layers, batch_first=True, dtype=torch.float64, bidirectional=False)
        self.decoder_layer = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True, dtype=torch.float64)
        self.embedding_output = nn.Embedding(29, n_hidden, dtype=torch.float64)

        self.hidden2query = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
        self.encoder2value = nn.Linear(29, n_hidden, dtype=torch.float64)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, 29, dtype=torch.float64)
        self.fc1 = nn.Linear(2 * n_hidden, 29, dtype=torch.float64)

        self.to(device)

    def encoder(self, x):

        # longueur_sequence = mask.sum(dim=1)
        # packed_input = pack_padded_sequence(x, longueur_sequence.cpu(), batch_first=True, enforce_sorted=False)

        # Encodeur
        out, (hn, c) = self.encoder_layer(x)
        # out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.fc(out)
        return out, hn, c

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

    def decoderWithAttn(self, encoder_outs, hidden, cell, target):
        # Décodeur avec attention
        # Initialisation des variables
        max_len = self.max_len # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch

        vec_in = torch.full((batch_size, 1), fill_value = self.symbol_to_int[self.start_symbol]).to(self.device)
        vec_out = torch.zeros((batch_size, max_len, 29)).to(self.device)  # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            out, (hidden,cell) = self.decoder_layer(self.embedding_output(vec_in), (hidden, cell))
            # # sans attention
            # out = self.fc(out)
            #
            # vec_in = torch.argmax(out, dim=2)
            #
            # vec_out[:, i] = out[:, 0]

            # avec attention
            attention_out, attention_weigths = self.attentionModule(out, encoder_outs)
            vec_in = torch.cat((out, attention_out), dim=2)
            vec_in = self.fc1(vec_in)
            vec_out[:, i] = vec_in[:, 0]

            # vec_in = torch.argmax(vec_in, dim=2)

            # Teaching forcing
            rand_val = torch.rand(1, device=self.device).item()
            if rand_val < self.teaching_forcing_ratio:
                vec_in = target[:, i].unsqueeze(1)
            else:
                vec_in = torch.argmax(out, dim=2)

        return vec_out, hidden, attention_weigths

    def forward(self, x, target):
        # Passe avant
        out, h, cell = self.encoder(x)

        out, hidden, attn = self.decoderWithAttn(out, h, cell, target)

        return out, h, attn

