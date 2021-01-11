import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    supported_rnns = {'rnn': nn.RNN,
                      'lstm': nn.LSTM,
                      'gru': nn.GRU
                      }

    def __init__(self,
                 input_size : int = 80, # number of mel
                 num_classes : int = 512,
                 num_rnn_layers = 3,
                 num_hidden_dim = 256,
                 dropout: float = 0.3,
                 bidirectional : bool = True,
                 rnn_type: str = 'lstm'
                 ) -> None :

        super(Encoder,self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_rnn_layers = num_rnn_layers
        self.num_hidden_dim = num_hidden_dim
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=num_hidden_dim,
            num_layers=num_rnn_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self,
                inputs: Tensor) -> Tensor:  # input = BxSxN
        output = self.rnn(inputs)

        return output




