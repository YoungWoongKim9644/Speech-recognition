import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):

    def __init__(self,mode = 'dot'):
        super(Attention,self).__init__()
        self.mode= mode

    def forward(self,
                decoder_state,  #shape: BxSxH
                encoder_output  #shape: BxSxH
               ):
        if self.mode == 'dot':
            attn_score = torch.bmm(decoder_state, encoder_output.transpose(1, 2)) #shape: BxSxS
            attn_distribution = F.softmax(attn_score, dim=2)
            context = torch.mm(attn_distribution, encoder_output)

            return context, attn_score

        #TODO 다른 attention 구현하기