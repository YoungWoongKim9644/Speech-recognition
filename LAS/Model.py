import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LAS(nn.Module):

    def __init__(self, encoder, decoder, device):
        super(LAS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing = 0.5

    def forward(self,
                encoder_inputs:Tensor,
                decoder_inputs:Tensor,
                teacher_forching_rate):
        teacher_forching_rate = self.teacher_forcing

        encoder_outputs = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, teacher_forching_rate)

        return decoder_outputs