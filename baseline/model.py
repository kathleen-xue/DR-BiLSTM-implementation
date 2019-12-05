import torch.nn as nn
import torch

class Bowman(nn.Module):
    def __init__(self, vocab, premise_emb=300, hypothesis_emb=300, premise_d=100, hypothesis_d=100, lstm_layers=1, dropout=0.1):
        super(Bowman, self).__init__()
        # vocab - vocab built for corpus
        # premise_emb - word embedding size for tokens in premise
        # hypothesis_emb - word embedding size for tokens in hypothesis
        # premise_d - sentence embedding size for premise
        # hypothesis_d - sentence embedding size for hypothesis
        # lstm_layers - layer number for LSTM model
        # dropout - dropout rate for the model
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.dropout = nn.Dropout(dropout)
        self.Premise_Enc = nn.LSTM(input_size=premise_emb, hidden_size=premise_d, num_layers=lstm_layers, batch_first=True)
        self.Hypothesis_Enc = nn.LSTM(input_size=hypothesis_emb, hidden_size=hypothesis_d, num_layers=lstm_layers, batch_first=True)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(premise_d + hypothesis_d, 3) # batch_size x 3

    def forward(self, premise_seq, hypothesis_seq):
        premise_seq = self.embedding(premise_seq) # batch_size x seq_len -> batch_size x seq_len x 300
        hypothesis_seq = self.embedding(hypothesis_seq) # batch_size x seq_len -> batch_size x seq_len x 300
        premise_seq = self.dropout(premise_seq)
        hypothesis_seq = self.dropout(hypothesis_seq)

        premise_output, _  = self.Premise_Enc(premise_seq) # batch_size x seq_len x 300 -> batch_size x seq_len x 100
        hypothesis_output, _  = self.Hypothesis_Enc(hypothesis_seq) # batch_size x seq_len x 300 -> batch_size x seq_len x 100
        premise_output = torch.mean(premise_output, 1) # batch_size x seq_len x 100 -> batch_size x 100
        hypothesis_output = torch.mean(hypothesis_output, 1) # batch_size x seq_len x 100 -> batch_size x 100
        next_in = torch.cat((premise_output, hypothesis_output), 1)  # [batch_size x 100, batch_size x 100] -> batch_size x 200
        #next_in = torch.cat((premise_output[ :, -1, :],hypothesis_output[ :, -1, :]), 1)
        next_in = self.dropout(next_in)
        tanh_out = self.tanh(self.tanh(self.tanh(next_in)))
        output = self.out(tanh_out) # batch_size x 200 -> batch_size x 3
        return torch.log_softmax(output, dim=1)