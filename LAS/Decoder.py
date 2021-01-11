import torch
import torch.nn as nn
import torch.nn.functional as F
import Attention
from torch import Tensor


class Decoder(nn.Module):
    supported_rnns ={'rnn': nn.RNN,
                     'lstm': nn.LSTM,
                    'gru': nn.GRU
                     }

    def __init__(self,
                 embedding_dim: int, # number of mel
                 num_classes : int = 512,
                 num_rnn_layers = 3,
                 num_hidden_dim = 256,
                 dropout: float = 0.3,
                 bidirectional : bool = False,
                 rnn_type: str = 'lstm',
                 pad_id: int = 0,
                 sos_id: int = 1,
                 eos_id: int = 2
                 ) -> None :

        super(Decoder, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.embedding = nn.Sequential(nn.Embedding(num_classes, embedding_dim),
                                       nn.Dropout(p=dropout)
                                       )
        self.num_classes = num_classes
        self.num_rnn_layers = num_rnn_layers
        self.num_hidden_dim = num_hidden_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = num_hidden_dim << 1 if bidirectional else num_hidden_dim
        self.attention = Attention()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.rnn = rnn_cell(
            input_size=embedding_dim,
            hidden_size=num_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(self.output_size, num_classes)

    def forward(self,
                inputs: Tensor,  # input = BxS ? BxSxN
                last_hidden: Tensor,  # hidden = num_layers * num_direction x B x H
                encoder_outputs,
                teacher_forcing_ratio=1.0
                ) -> Tensor:

        inputs = inputs.transpose(1, 2)  # BxSxI -> SxBxI
        seq_len = inputs.size()[0]
        embedded = self.embedding(inputs)  # embedded shape: BxSxE_dim
        rnn_output, hidden_state = self.rnn(embedded, last_hidden)  # output shape= SxBxO

        context, attn_score = self.attention(rnn_output, encoder_outputs)
        context = torch.cat((rnn_output, context), dim=2)
        context = context.view(-1, self.num_hidden_dim)

        output = self.fc(torch.tanh(context))
        output = F.log_softmax(output, dim=1)  # output shape : B*Sx

        return output


"""

    def forward(self,
                decoder_inputs: Tensor,
                encoder_output: Tensor,
                teacher_forching_ratio: float = 1.0
                ) -> dict:

        seq_len = decoder_inputs.size()[1]  # input_data shape : (batch, seq_len)
        embedded = self.embedding(decoder_inputs)  # embedded shape : (batch, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        rnn_output, _ = self.rnn(embedded)  # rnn_output shape : (batch, seq_len, hidden_size)

        context_vector = self.attention(rnn_output, encoder_output, encoder_output)
        context_vector = torch.cat((context_vector, rnn_output), dim=2)  # shape : (batch, seq_len, hidden_size << 1)

        context_vector = context_vector.view(-1, self.hidden_size << 1).contiguous()
        output = self.fc(context_vector)  # output shape : (batch * seq_len, num_vocabs)
        output = F.log_softmax(output, dim=1)

        return output, seq_len
        
    def forward_step(self,
                     inputs: Tensor,  # input = BxS ? BxSxN
                     last_hidden: Tensor,  # hidden = num_layers * num_direction x B x H
                     encoder_outputs,
                     teacher_forcing_ratio=1.0
                     ) -> Tensor:

        inputs = inputs.transpose(1, 2)  # BxSxI -> SxBxI
        seq_len = inputs.size()[0]
        embedded = self.embedding(inputs)
        rnn_output, hidden_state = self.rnn(embedded, last_hidden)  # output = SxBxO
        context, attn_score = self.attention(rnn_output, encoder_outputs)
        context = torch.cat((rnn_output, context), dim=2)
        #output = output.transpose(1, 2)  # fc : (B , * , O)
        output = self.fc(torch.tanh(context))
        step_output = F.log_softmax(output, dim=1) # output shape : (batch*seq_len, num_vocabs)

"""