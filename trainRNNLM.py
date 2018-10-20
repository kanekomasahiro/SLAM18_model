from __future__ import division
import onmt
import BLSTM
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import argparse
import random
from collections import defaultdict

parser = argparse.ArgumentParser(description='train')

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model/model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-all_dicts', default='data/dicts_for_all', type=str,
                    help="""This dict is used for all GAN's process.""")

## Model options

parser.add_argument('-layers', type=int, default=1,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=400,
                    help='Size of LSTM hidden states')
parser.add_argument('-extra_hidden_size', type=int, default=50,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_size', type=int, default=100,
                    help='Word embedding sizes')
parser.add_argument('-user_size', type=int, default=50,
                    help='Word embedding sizes')
parser.add_argument('-session_size', type=int, default=20,
                    help='Word embedding sizes')
parser.add_argument('-format_size', type=int, default=20,
                    help='Word embedding sizes')
parser.add_argument('-pos_size', type=int, default=20,
                    help='Word embedding sizes')
parser.add_argument('-bptt', type=int, default=18,
                    help='Word embedding sizes')
parser.add_argument('-brnn', action='store_false',
                    help='Use a bidirectional encoder')
parser.add_argument('-time_step', action='store_true',
                    help='')

## Optimization options

parser.add_argument('-epochs', type=int, default=30,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-emb_init', action="store_true",
                    help="""aaa""")
#parser.add_argument('-optim', default='sgd',
#parser.add_argument('-optim', default='adam',
parser.add_argument('-optim', default='adadelta',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.2,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-seq_length', type=int, default=90,
                    help="Maximum sequence length")
#parser.add_argument('-stop_lr', type=int, default=0.005,
parser.add_argument('-stop_lr', type=int, default=0.000005,
                    help="Maximum sequence length")

#learning rate
parser.add_argument('-learning_rate', type=float, default=0.1,
#parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=20,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-gpus', default=[1], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

opt = parser.parse_args()

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def measure(pres, tags):
    #num_correct = pres.data.eq(tags.data).sum()
    num_correct = pres.data.eq(tags.data).masked_select(tags.ne(onmt.Constants.PAD).data).sum()
    #pres_pos = pres.data.eq(1)
    pres_pos = pres.data.eq(1).masked_select(tags.ne(onmt.Constants.PAD).data)
    #tags_pos = tags.data.eq(1)
    tags_pos = tags.data.eq(1).masked_select(tags.ne(onmt.Constants.PAD).data)
    num_pos = pres_pos.eq(tags_pos).masked_select(tags_pos.ne(0))

    return num_correct, num_pos.sum(), pres_pos.sum(), tags_pos.sum()


def Criterion(tagSize):
    weight = torch.ones(tagSize)
    weight[onmt.Constants.PAD] = 0
    #crit = nn.CrossEntropyLoss(weight, size_average=False)
    crit = nn.NLLLoss(weight, size_average=False)
    #crit = nn.CrossEntropyLoss(weight)
    if opt.gpus:
        crit.cuda()
    return crit


def eval(model, criterion, data, dicts, user_history):
    criterion = nn.CrossEntropyLoss(size_average=False)
    total_loss = 0
    total_words = 0
    num_pos, pres_pos, tags_pos = 0, 0, 0
    num_correct = 0

    model.eval()
    for i in range(len(data)):
        batch = data[i]
        for input, tag, days, session, format, time, pos in zip(batch['token'], batch['gold'], batch['days'], batch['session'], batch['format'], batch['time'], batch['pos']):
            data_size = input.size(1)
            user = Variable(batch['user'][0][:,:data_size].data)
            near_days = []
            for j in range(data_size):
                for l in sorted(user_history[user.data[:,j][0]].items()):
                    if l[0] <= days.data[:,j][0]:
                        near_day = l[0]
                    else:
                        break
                near_days.append(near_day)
            history = torch.stack([user_history[user.data[:,j].cpu()[0]][near_days[j]][0] for j in range(data_size)])
            history = Variable(history, volatile=True).cuda()
            history_hidden0 = torch.stack([user_history[user.data[:,j].cpu()[0]][near_days[j]][1] for j in range(data_size)])
            history_hidden0 = history_hidden0.view(history_hidden0.size(1), history_hidden0.size(0), -1)
            history_hidden0 = Variable(history_hidden0, volatile=True).cuda()
            history_hidden1 = torch.stack([user_history[user.data[:,j].cpu()[0]][near_days[j]][2] for j in range(data_size)])
            history_hidden1 = history_hidden1.view(history_hidden1.size(1), history_hidden1.size(0), -1)
            history_hidden1 = Variable(history_hidden1, volatile=True).cuda()
            history_hidden = (history_hidden0, history_hidden1)

            outputs, history, history_hidden = model(input, tag, user, days, session, format, time, pos, history, history_hidden)
            tag = tag.view(-1)
            pad_t = Variable(torch.FloatTensor([0]*outputs.size(0)).unsqueeze(1)).cuda()
            pad_outputs = torch.cat((outputs, pad_t), 1)
            total_loss += criterion(pad_outputs, tag).data[0]

            #total_words += tag.data.size(0)
            total_words += tag.data.ne(onmt.Constants.PAD).sum()
            pres = torch.max(outputs, 1)[1]
            a, b, c, d = measure(pres, tag)
            num_correct += a
            num_pos += b
            pres_pos += c
            tags_pos += d
    presicion = num_pos / pres_pos if pres_pos != 0 else 0
    recall = num_pos / tags_pos if tags_pos != 0 else 0
    f_value = (2*presicion*recall)/(presicion+recall) if (presicion+recall) != 0 else 0
    model.train()
    return total_loss/total_words, num_correct / total_words, presicion, recall, f_value


def trainModel(model, trainData, validData, dataset, model_optim, dicts):
    model.train()
    # define criterion of each GPU
    criterion = Criterion(3)

    def preGen():
        #if opt.extra_shuffle and epoch > opt.curriculum:
            #trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))
        user_history = defaultdict(lambda: dict())

        total_loss, num_correct, total_words = 0, 0, 0
        num_pos, pres_pos, tags_pos = 0, 0, 0
        for i in range(len(trainData)):
            model.zero_grad()
            batchIdx = batchOrder[i]
            batch = trainData[batchIdx] # exclude original indices
            history = Variable(torch.FloatTensor([[0 for _ in range(opt.rnn_size)] for _ in range(batch['token'][0].size(1))]))
            history = history.cuda()
            history_hidden = None
            loss = 0
            cycle = 1
            batch_size = len(batch['token'])
            for input, tag, days, session, format, time, pos in zip(batch['token'], batch['gold'], batch['days'], batch['session'], batch['format'], batch['time'], batch['pos']):
                data_size = input.size(1)
                if history_hidden != None:
                    history = history[:data_size,:]
                    history_hidden = (history_hidden[0][:,:data_size,:], history_hidden[1][:,:data_size,:])
                user = Variable(batch['user'][0][:,:data_size].data)
                outputs, history, history_hidden = model(input, tag, user, days, session, format, time, pos, history, history_hidden)
                tag = tag.view(-1)
                pad_t = Variable(torch.FloatTensor([0]*outputs.size(0)).unsqueeze(1)).cuda()
                pad_outputs = torch.cat((outputs, pad_t), 1)
                loss += criterion(pad_outputs, tag)

                #total_words += tag.data.size(0)
                total_words += tag.data.ne(onmt.Constants.PAD).sum()
                pres = torch.max(outputs, 1)[1]
                a, b, c, d = measure(pres, tag)
                num_correct += a
                num_pos += b
                pres_pos += c
                tags_pos += d
                if cycle%opt.bptt == 0 or cycle == batch_size: #ここのbatch_sizeの箇所確かめたい
                    # loss of tag's predict
                    loss.backward()
                    model_optim.step()
                    model.zero_grad()
                    total_loss += loss.data[0]
                    loss = 0
                    history = Variable(history.data)
                    history_hidden = (Variable(history_hidden[0].data), Variable(history_hidden[1].data))
                cycle += 1

                for j in range(data_size):
                    user_history[user.data[:,j][0]][days.data[:,j][0]] = [history.data[j].cpu(), history_hidden[0].data[:,j].cpu(), history_hidden[1].data[:,j].cpu()]
        presicion = num_pos / pres_pos if pres_pos != 0 else 0
        recall = num_pos / tags_pos if tags_pos != 0 else 0
        f_value = (2*presicion*recall)/(presicion+recall) if (presicion+recall) != 0 else 0
        return total_loss/total_words, num_correct/total_words, presicion, recall, f_value, dict(user_history)
    
    print('Start Generator Pretraining')
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        #  (1) train for one epoch on the training set
        print("epoch{}:".format(epoch))
        train_loss, acc, pre, re, fv, user_history = preGen()
        print('log loss:{} acc:{} precision:{} recall:{} fv:{}'.format(train_loss, acc*100, pre*100, re*100, fv*100))
        model_state_dict = model.state_dict()
        valid_loss, valid_acc, valid_pre, valid_re, valid_fv = eval(model, criterion, validData, dicts, user_history)
        print('val_loss:{} val_acc:{} val_pre:{} val_re{} val_fv:{}'.format(valid_loss, valid_acc*100, valid_pre*100, valid_re*100, valid_fv*100))
        #model_optim.updateLearningRate(valid_loss, epoch)
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dataset['dicts'],
            'type': dataset['train']['type'],
            'features': dataset['features'],
            'opt': opt,
            'epoch': epoch,
            'model_optim': model_optim,
            'user_history': user_history,
        }
        torch.save(checkpoint,
                   '{}_acc_{:.2f}_epoch_{}.pt'.format(opt.save_model, 100*valid_fv, epoch))

def main():

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    features = dataset['features']

    trainData = onmt.Dataset(dataset['train']['data'], opt.gpus)

    validData = onmt.Dataset(dataset['valid']['data'], opt.gpus,
                             volatile=True)

    dicts = dataset['dicts']
    opt_pre = dataset['opt']
    print(' * vocabulary size. = %d' %
          (dicts['vocab'].size()))
    print(' * maximum batch size. %d' % opt_pre.batch_size)

    print('Building model...')

    model = BLSTM.BLSTM(opt, dicts, dataset['train']['type'], features)

    
    if opt.param_init:
        print('Intializing model parameters.')
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)
        model.word_lut.weight.data[onmt.Constants.PAD].zero_()
        model.session_lut.weight.data[onmt.Constants.PAD].zero_()
        model.format_lut.weight.data[onmt.Constants.PAD].zero_()
        model.pos_lut.weight.data[onmt.Constants.PAD].zero_()
    if opt.emb_init:
        print('Intializing embeddings.')
        w2v = dicts['w2v']
        for i in range(model.word_lut.weight.size(0)):
            if i in w2v:
                model.word_lut.weight[i].data.copy_(torch.FloatTensor(w2v[i]))
                #model.word_lut.weight[i].data.copy_(torch.from_numpy(w2v[i]))

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()

    model_optim = onmt.Optim(
        opt.optim, opt.learning_rate, opt.max_grad_norm, opt.stop_lr,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at
    )
    model_optim.set_parameters(model.parameters())
    trainModel(model, trainData, validData, dataset, model_optim, dicts)


if __name__ == "__main__":
    main()
