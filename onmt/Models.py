import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from math import ceil
import gc

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size
        self.vocab_size = dicts.size()

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(self.vocab_size,
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            emb = pack(self.word_lut(input[0]), input[1])
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        self.input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  self.input_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, self.input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    #ここのmax_length
    def forward(self, input, hidden, context, init_output, max_length=40):
        if input is not None:
            emb = self.word_lut(input)

            # n.b. you can increase performance if you compute W_ih * x for all
            # iterations in parallel, but that's only possible if
            # self.input_feed=False
            outputs = []
            output = init_output
            for emb_t in emb.split(1):
                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)
                output, hidden = self.rnn(emb_t, hidden)
                output, attn = self.attn(output, context.t())
                output = self.dropout(output)
                outputs += [output]

            outputs = torch.stack(outputs)
            return outputs, hidden, attn
        else:
            #decoderの入力のための開始記号
            input = Variable(torch.LongTensor.zero_(torch.LongTensor(1, len(init_output)))).cuda() + 2
            emb = self.word_lut(input)
            emb = emb.split(1)[0]
            outputs = []
            output = init_output
            for _ in range(max_length):
                emb_t = emb.squeeze(0)
                #if self.input_feed:
                    #emb_t = torch.cat([emb_t, output], 1)
                output, hidden = self.rnn(emb_t, hidden)
                output, attn = self.attn(output, context.t())
                output = self.dropout(output)
                emb = output
                #if [output] == []:
                    #break
                outputs += [output]
            outputs = torch.stack(outputs)
            return outputs, hidden, attn

    def decode_one_step(self, input, hidden, context):
        emb = self.word_lut(input)
        emb_t = emb.view(-1, self.input_size)
        output, hidden = self.rnn(emb_t, hidden)
        output, attn = self.attn(output, context.t())
        output = self.dropout(output)
        
        return output, hidden

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.softmax = nn.Softmax()
        self.softmax = nn.Sigmoid()
        self.vocab_size = self.encoder.vocab_size

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpos_e(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        if input[1] is not None:
            tgt = input[1][:-1]  # exclude last target from inputs
        else:
            #decoderの時に答えを入力に使わない時
            tgt = None
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)
        return out

    def get_rewards(self, samples, batch, dis, generator, rollout_time=1):
        batch_size = samples.size(1)
        sequence_length = samples.size(0)
        
        reward_mat = torch.zeros(batch_size, sequence_length).cuda()
        for given in range(1, sequence_length):
            rewards = self.roll_out(batch, given, dis, rollout_time, generator, sequence_length, batch_size, samples)
            reward_mat[:, given-1] = rewards
        reward_mat[:, sequence_length-1] = dis.get_reward(samples.t(), batch.t())
        return reward_mat

    def roll_out(self, batch, given, dis, rollout_time, generator, sequence_length, batch_size, samples):
        enc_hidden, context = self.encoder(batch)
        init_output = self.make_init_decoder_output(context)
        preds, dec_hidden, _attn = self.decoder(None, enc_hidden, context, init_output, max_length=given)
        preds = preds[given-1]
        preds = generator(preds)

        rewards = []
        gen_x = samples.data.cuda().t()
        for _ in range(rollout_time):
            hidden = dec_hidden
            hidden = (self._fix_enc_hidden(hidden[0]),
                          self._fix_enc_hidden(hidden[1]))
            scores = preds
            for i in range(given, sequence_length):
                generated = scores.data.cuda()
                generated = [torch.multinomial(generated[j], 1) for j in range(batch_size)]
                generated = torch.stack(generated)
                gen_x[:,i] = generated
                x = Variable(generated, volatile=True)
                scores, hidden = self.decoder.decode_one_step(x, hidden, context)
            gen_x = Variable(gen_x, volatile=True)
            rewards.append(dis.get_reward(gen_x, batch.t()))
        rewards = torch.stack(rewards)
        return torch.mean(rewards, 0)


class CNN(nn.Module):
    
    def __init__(self, opt, dicts):
        self.max_leng = opt.seq_length
        self.hidden_sizes = opt.cnn_size
        self.dropout_switches = opt.cnn_dropout_swithces
        super(CNN, self).__init__()
        self.word_lut = nn.Embedding(dicts['all'].size(),
                                opt.word_vec_size,
                                padding_idx=onmt.Constants.PAD)
        self.conv_1 = nn.Conv2d(in_channels=opt.word_vec_size*2,
                                out_channels=opt.num_filters,
                                kernel_size=opt.filter_sizes[0])
        self.pool = nn.MaxPool2d(3, 3)
        self.conv_2 = nn.Conv2d(in_channels=opt.num_filters,
                                out_channels=opt.num_filters,
                                kernel_size=opt.filter_sizes[0])
        #文のmax sizeに対して+2(for padding)i:開始記号で+1
        ins = int((int(((self.max_leng + 2)-opt.filter_sizes[0])/3)-opt.filter_sizes[0])/3)**2*opt.num_filters
        self.hid_layer = nn.Linear(ins,20)
        self.logistic = nn.Linear(20,2)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, input):
        #at least one padding on both side
        #ここ工夫したい
        input_s = input[1]
        pad_size1 = int(((self.max_leng + 2)-input_s.size(1))/2)
        pad_size2 = int(ceil(((self.max_leng + 2)-input_s.size(1))/2))
        if pad_size1 > 0:
            leftPad = torch.LongTensor.zero_(torch.LongTensor(input_s.size(0), pad_size1)).cuda()
            leftPad = Variable(leftPad)
            input_s = torch.cat((leftPad,input_s),1)
        if pad_size2 > 0:
            rightPad = torch.LongTensor.zero_(torch.LongTensor(input_s.size(0), pad_size2)).cuda()
            rightPad = Variable(rightPad)
            input_s = torch.cat((input_s, rightPad),1)
        
        input_t = input[0]
        pad_size1 = int(((self.max_leng + 2)-input_t.size(1))/2)
        pad_size2 = int(ceil(((self.max_leng + 2)-input_t.size(1))/2))
        if pad_size1 > 0:
            leftPad = torch.LongTensor.zero_(torch.LongTensor(input_t.size(0), pad_size1)).cuda()
            leftPad = Variable(leftPad)
            input_t = torch.cat((leftPad,input_t),1)
        if pad_size2 > 0:
            rightPad = torch.LongTensor.zero_(torch.LongTensor(input_t.size(0), pad_size2)).cuda()
            rightPad = Variable(rightPad)
            input_t = torch.cat((input_t, rightPad),1)

        emb_s = self.word_lut(input_s)
        emb_t = self.word_lut(input_t)
        t_broadcast = emb_t.repeat(emb_s.size(1),1,1,1)
        s_broadcast = emb_s.repeat(emb_t.size(1),1,1,1)
        t_broadcast = t_broadcast.view(emb_t.size(0),emb_s.size(1),emb_t.size(1),-1)
        s_broadcast = s_broadcast.view(emb_s.size(0),emb_s.size(1),emb_t.size(1),-1)
        x_2D = torch.cat((t_broadcast, s_broadcast), 3)
        x_2D = x_2D.view(x_2D.size(0),x_2D.size(3),x_2D.size(1),-1)
        x_2D = self.pool(F.tanh(self.conv_1(x_2D)))
        hid_in = self.pool(F.tanh(self.conv_2(x_2D)))
        hid_in = hid_in.view(hid_in.size(0),-1)
        hid_in = F.tanh(self.hid_layer(hid_in))
        if self.dropout_switches[0]:
            hid_in = self.dropout(hid_in)
        hid_out = self.logistic(hid_in)
        pred_prob = self.softmax(hid_out)
        return pred_prob

    def get_reward(self, x_input, batch):
        pred = self.forward((x_input, batch)).data.cuda()
        return pred[:,1]
