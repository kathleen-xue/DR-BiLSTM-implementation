"""
Definition of the model.
"""
# Aurelien Coet, 2018.
import numpy as np
import torch.nn as nn
import torch
import os
import zipfile
import pickle
import fnmatch
import json
import string

from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
from tqdm import tqdm
from drlstm.utils import correct_predictions


from torch.utils.data import DataLoader
from drlstm.data import NLIDataset

import matplotlib.pyplot as plt

from drlstm.layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from drlstm.utils import get_mask, replace_masked

class DRLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):

        super(DRLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self.debug = False

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)
        #print ('embedding_dim: ')
        #print (embedding_dim)
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)
        
        self._encoding1 = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        int(self.hidden_size/2),
                                        bidirectional=True)
        self._encoding2 = Seq2SeqEncoder(nn.LSTM,
                                        self.hidden_size,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)
        
        self._composition1 = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)
        
        self._composition2 = Seq2SeqEncoder(nn.LSTM,
                                           2*self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_model_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):

        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        if self.debug:
          print (embedded_premises.size()) # 32,61,300
          print (embedded_hypotheses.size()) # 32,57,300
        #if self.dropout:
        #    embedded_premises = self._rnn_dropout(embedded_premises)
        #    embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)


        
        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)
        
        
        encoded_premises1 = self._encoding1(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses1 = self._encoding1(embedded_hypotheses,
                                            hypotheses_lengths)
        
        if self.debug:
          print ('encoded_premises1')
          print (encoded_premises1.size())
          print ('encoded_hypo1')
          print (encoded_hypotheses1.size())

        encoded_premises2 = self._encoding2(encoded_hypotheses1,
                                          hypotheses_lengths)
        encoded_hypotheses2 = self._encoding2(encoded_premises1,
                                            premises_lengths)
        
        if self.debug:
          print ('encoded_premises2')
          print (encoded_premises2.size())
          print ('encoded_hypo2')
          print (encoded_hypotheses2.size())
        
        """
        encoded_premises1
        torch.Size([32, 36, 300])
        encoded_hypo1
        torch.Size([32, 22, 300])
        encoded_premises2
        torch.Size([32, 22, 600])
        encoded_hypo2
        torch.Size([32, 36, 600])
        #print (premises_lengths.size()) # 32
        #print (hypotheses_lengths.size()) # 32
        
        #print (encoded_premises.size()) # 32,36,600
        #print (encoded_hypotheses.size()) # 32,22,600

        """

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises2, hypotheses_mask,
                            encoded_hypotheses2, premises_mask)
        
        if self.debug:
          print ('after attention')
          print ('attended_premises: ')
          print (attended_premises.size())
          print ('attended_hypotheses: ')
          print (attended_hypotheses.size())

        enhanced_premises = torch.cat([encoded_premises2,
                                       attended_premises,
                                       encoded_premises2 - attended_premises,
                                       encoded_premises2 * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses2,
                                         attended_hypotheses,
                                         encoded_hypotheses2 -
                                         attended_hypotheses,
                                         encoded_hypotheses2 *
                                         attended_hypotheses],
                                        dim=-1)
       
        if self.debug:
          print ('enhanced_premises: ')
          print (enhanced_premises.size())
          print ('enhanced_hypotheses: ')
          print (enhanced_hypotheses.size())
        
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)
        
        if self.debug:
          print ('projected_premises:')
          print (projected_premises.size())
          print ('projected_hypotheses:')
          print (projected_hypotheses.size())

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        if self.debug:
          print ('projected_premises after dropout:')
          print (projected_premises.size())
          print ('projected_hypotheses after dropout:')
          print (projected_hypotheses.size())
        """
        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)
        print ('v_ai:')
        print (v_ai.size())
        print ('v_bj')
        print (v_bj.size())
        """

        v_ai1 = self._composition1(projected_premises, hypotheses_lengths)
        v_bj1 = self._composition1(projected_hypotheses, premises_lengths)

        if self.debug:
          print ('v_ai1')
          print (v_ai1.size())
          print ('v_bj1')
          print (v_bj1.size())

        v_ai2 = self._composition2(v_ai1, hypotheses_lengths)
        v_bj2 = self._composition2(v_bj1, premises_lengths)

        if self.debug:
          print ('v_ai2')
          print (v_ai2.size())
          print ('v_bj2')
          print (v_bj2.size())

        
        # max pooling
        v_ai = torch.max(v_ai1, v_ai2)
        v_bj = torch.max(v_bj1, v_bj2)

        if self.debug:
          print ('v_ai after max')
          print (v_ai.size())
          print ('v_bj after max')
          print (v_bj.size())

        v_a_avg = torch.sum(v_bj * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_ai * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_bj, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_ai, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        if self.debug:
          print ('v:')
          print (v.size())
        logits = self._classification(v)
        if self.debug:
          print ('logits:')
          print (logits.size())
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


def _init_model_weights(module):
    """
    Initialise the weights of the model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
