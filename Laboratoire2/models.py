# GRO722 Laboratoire 2
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
from base64 import encode

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import embedding
from triton.ops import attention


class Seq2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True, bidirectional=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        # Encodeur

        # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------

        out, hidden = self.encoder_layer(self.fr_embedding(x))
        out = self.fc(out)
        # ---------------------- Laboratoire 2 - Question 3 - Fin de la section à compléter -----------------

        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage
        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------
            out, hidden = self.decoder_layer(self.en_embedding(vec_in), hidden)
            out = self.fc(out)
            vec_in = torch.argmax(out,dim=2)
            vec_out[:,i] = out[:, 0]
            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------
        return vec_out, hidden, None

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn


class Seq2seq_attn(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq_attn, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2*n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)
        self.encoder2value = nn.Linear(self.dict_size['en'], n_hidden)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        #Encodeur

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        out, hidden = self.encoder_layer(self.fr_embedding(x))
        out = self.fc(out)
        
        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return out, hidden

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)
        values = self.encoder2value(values)

        # Attention

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
        similarity = torch.bmm(query, values.transpose(1,2))
        attention_weights = torch.nn.functional.softmax(similarity, dim=2)
        attention_output = torch.bmm(attention_weights, values)
        

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        # Décodeur avec attention
        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage
        attention_weights = torch.zeros((batch_size, self.max_len['fr'], self.max_len['en'])).to(self.device) # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            out, hidden = self.decoder_layer(self.en_embedding(vec_in), hidden)
            attention_out, weights = self.attentionModule(out,encoder_outs)
            vec_in = self.att_combine(torch.cat((out, attention_out),dim=2))
            vec_in = self.fc(vec_in)
            vec_out[:,i] = vec_in[:,0]
            vec_in = torch.argmax(out, dim=2)
            attention_weights[:,:,i] = weights[:,0]
        return vec_out, hidden, attention_weights


    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out,h)
        return out, hidden, attn
