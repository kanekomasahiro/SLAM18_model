import torch.nn.functional as F
import onmt
import torch
import torch.nn as nn
from torch.autograd import Variable

class BLSTM(nn.Module):

    def __init__(self, opt, dicts, types, features=None):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_size + opt.user_size + opt.session_size + opt.format_size + opt.rnn_size + opt.pos_size + 1 + 1

        super(BLSTM, self).__init__()
        self.word_lut = nn.Embedding(dicts['vocab'].size(),
                                  opt.word_size,
                                  padding_idx=onmt.Constants.PAD)
        self.user_lut = nn.Embedding(dicts['feature']['user'].size(),
                                  opt.user_size,
                                  padding_idx=onmt.Constants.PAD)
        self.session_lut = nn.Embedding(dicts['feature']['session'].size(),
                                  opt.session_size,
                                  padding_idx=onmt.Constants.PAD)
        self.format_lut = nn.Embedding(dicts['feature']['format'].size(),
                                  opt.format_size,
                                  padding_idx=onmt.Constants.PAD)
        self.pos_lut = nn.Embedding(dicts['feature']['pos'].size(),
                                  opt.pos_size,
                                  padding_idx=onmt.Constants.PAD)
        self.history_rnn_f = nn.LSTM(opt.rnn_size+1, opt.rnn_size,
                    num_layers=opt.layers)
        self.history_rnn_b = nn.LSTM(opt.rnn_size+1, opt.rnn_size,
                    num_layers=opt.layers)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                    num_layers=opt.layers,
                    #dropout=opt.dropout,
                    bidirectional=opt.brnn)
        self.e_hidden = nn.Linear(opt.rnn_size, opt.extra_hidden_size)
        self.output = nn.Linear(opt.extra_hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.outputDropout = nn.Dropout(p = opt.dropout)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def forward(self, input, gold, user, days, session, format, time, pos, history, history_hidden, test=False):
        days = days.t()
        time = time.t()

        word_emb = self.word_lut(input)
        user_emb = self.user_lut(user)
        session_emb = self.session_lut(session)
        format_emb = self.format_lut(format)
        pos_emb = self.pos_lut(pos)

        history = torch.stack([history for _ in range(word_emb.size(0))])
        user_emb = torch.stack([user_emb for _ in range(word_emb.size(0))]).squeeze(1)
        session_emb = torch.stack([session_emb for _ in range(word_emb.size(0))]).squeeze(1)
        format_emb = torch.stack([format_emb for _ in range(word_emb.size(0))]).squeeze(1)
        days = torch.stack([days for _ in range(word_emb.size(0))])
        time = torch.stack([time for _ in range(word_emb.size(0))])

        emb = torch.cat((word_emb, user_emb, session_emb, format_emb, history, days, time, pos_emb), 2)
        outputs, hidden_t = self.rnn(emb)
        hidden_t = (self._fix_enc_hidden(hidden_t[0]),
                      self._fix_enc_hidden(hidden_t[1]))
        if not test:
            gold = gold.unsqueeze(2).float()
            history = torch.cat((outputs, gold), 2)
            history_f, history_hidden_f = self.history_rnn_f(history, history_hidden)

            idx = [i for i in range(history.size(0)-1, -1, -1)]
            idx = Variable(torch.LongTensor(idx)).cuda()
            history = history.index_select(0, idx)

            history_b, history_hidden_b = self.history_rnn_b(history, history_hidden)
            #history = torch.cat((history_f[-1], history_b[-1]), dim=1)
            #history_hidden = (history_hidden_f[0]+history_hidden_b[0], history_hidden_f[1]+history_hidden_b[1])
            history = torch.mean(torch.stack((history_f[-1], history_b[-1])), dim=0)
            history_hidden = (torch.mean(torch.cat((history_hidden_f[0], history_hidden_b[0])), dim=0).unsqueeze(0), torch.mean(torch.cat((history_hidden_f[1], history_hidden_b[1])), dim=0).unsqueeze(0))
            #history = history[-1]

        outputs = outputs.view(-1, self.hidden_size*2) if self.num_directions == 2 else outputs.view(-1, self.hidden_size)
        outputs = F.relu(self.e_hidden(outputs))
        #outputs = self.output(outputs)
        outputs = self.output(self.outputDropout(outputs))
        if not test:
            pres = self.log_softmax(outputs)
        else:
            pres = self.softmax(outputs)
        
        return pres, history, history_hidden
