# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1, type="elman"):
        super(Model, self).__init__()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------

        # Création d'un RNN Elman
        if type == "elman":
            self.rnn = nn.RNN(1, n_hidden, n_layers, batch_first=True)

        elif type == "lstm":
            self.rnn = nn.LSTM(1, n_hidden, n_layers, batch_first=True)
        elif type == "gru":
             self.rnn = nn.GRU(1, n_hidden, n_layers, batch_first=True)        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
        self.lin = nn.Linear(25, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, h=None):
        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        # Passe avant du RNN
        if h is None:
            x, h = self.rnn(x)

        else:
            x, h = self.rnn(x, h)

        # Flatten the x tensor
        x = self.lin(x)
        x = self.tanh(x)
        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------

        return x, h

if __name__ == '__main__':
    x = torch.zeros((100,2,1)).float()
    model = Model(25)
    output,h = model(x)
    print(output.shape)
    print(h.shape)
