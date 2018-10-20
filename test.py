from __future__ import division

import Testor
import onmt
import torch
import argparse
import math
from nltk import word_tokenize
from collections import defaultdict
from torch.autograd import Variable

import pickle

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-data',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-output', default='result/dev.pre',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-batch_size', type=int, default=1,
                    help="")
parser.add_argument('-gpu', type=int, default=4,
                    help="Device to run on")


def changeNumToZero(num_str):
    try:
        float(num_str)
        return '0'
    except ValueError:
        return num_str


def changeStrToFloat(num_str):
    try:
        return float(num_str)
    except ValueError:
        return num_str


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


def feature_to_id(datas, dicts, gpu):
    new_datas = {}
    keys = list(datas.keys())
    for key, value in datas.items():
        new_datas[key] = defaultdict(lambda: list())

    for key, value in datas.items():
        new_datas[key]['user'] += [torch.LongTensor(dicts['feature']['user'].convertToIdx([key], onmt.Constants.UNK_WORD))]
        for k, v in value.items():
            if k == 'token':
                for vv in v:
                    new_datas[key][k] += [torch.LongTensor(dicts['vocab'].convertToIdx(vv,
                                            onmt.Constants.UNK_WORD))]
            elif k == 'Dependency head' or k == 'gold':
                for vv in v:
                    new_datas[key][k] += [torch.LongTensor(vv)]
            else:
                if k in dicts['feature']:
                    for vv in v:
                        if isinstance(vv, list):
                            new_datas[key][k] += [torch.LongTensor(dicts['feature'][k].convertToIdx(vv,
                                                    onmt.Constants.UNK_WORD))]
                        else:
                            new_datas[key][k] += [torch.LongTensor(dicts['feature'][k].convertToIdx([vv],
                                                    onmt.Constants.UNK_WORD))]
                else:
                    for vv in v:
                        if type(vv) == int:
                            new_datas[key][k] += [torch.LongTensor([vv])]
                        elif type(vv) == float:
                            new_datas[key][k] += [torch.FloatTensor([vv])]
                if k == 'uniqueID':
                    new_datas[key]['id'] += v

    for key, value in datas.items():
        new_datas[key] = dict(new_datas[key])
        new_datas[key]['size'] = len(new_datas[key]['token'])

    return new_datas


def main():
    types = {'user':list(), 'countries':list(), 'client':list(), 'session':list(), 'format':list(), 'uniqueID':list(), 'vocab':dict(), 'pos':list(), 'Morphological':list(), 'Dependency label':list()}
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    model = Testor.Testor(opt)

    outF = open(opt.output, 'w')

    datas = []
    result = []

    count = 0
    
    sentence_property = {}
    user_datas = defaultdict(lambda: defaultdict(lambda: list()))
    for d in open(opt.data):
        d = d.strip()
        if d:
            if d[0] == '#':
                tokens = []
                uniqids = []
                golds = []
                poses = []
                morphologicals = []
                dependencyLabels = []
                dependencyHeads = []

                d = d.split()
                d.pop(0)
                for metadata in d:
                    metadatas = metadata.split(':')
                    if metadatas[0] == 'time' and metadatas[1] == 'null':
                        sentence_property[metadatas[0]] = changeStrToFloat(-1)
                    elif metadatas[0] in types:
                        if metadatas[0] == 'client':
                            if metadatas[1] != 'web':
                                sentence_property[metadatas[0]] = 'mobile'
                                types[metadatas[0]].append('mobile')
                            else:
                                sentence_property[metadatas[0]] = 'web'
                                types[metadatas[0]].append('web')
                        else:
                            sentence_property[metadatas[0]] = metadatas[1]
                            types[metadatas[0]].append(metadatas[1])
                    else:
                        sentence_property[metadatas[0]] = changeStrToFloat(metadatas[1])
            else:
                d = d.split()
                user = sentence_property['user']
                for i, sen_data in enumerate(d):
                    if i == 0:
                        uniqids.append(sen_data)
                        types['uniqueID'].append(sen_data)
                        #if devKey:
                            #golds.append(keyD[sen_data])
                    elif i == 1:
                        w = changeNumToZero(sen_data).lower()
                        tokens.append(w)
                        if w in types['vocab']:
                            types['vocab'][w] += 1
                        else:
                            types['vocab'][w] = 1
                    elif i == 2:
                        poses.append(sen_data)
                        types['pos'].append(sen_data)
                    elif i == 3:
                        morphologicals.append(sen_data)
                        types['Morphological'].append(sen_data)
                    elif i == 4:
                        dependencyLabels.append(sen_data)
                        types['Dependency label'].append(sen_data)
                    elif i == 5:
                        dependencyHeads.append(int(sen_data))
                    elif i == 6:
                        golds.append(int(sen_data))
        else:
            #sentence_property['length'] = len(word_property['token'])
            #datas.append({'sentence': sentence_property, 'word': dict(word_property)})
            for key, value in sentence_property.items():
                if key != 'user':
                    user_datas[user][key].append(value)
            user_datas[user]['token'].append(tokens)
            user_datas[user]['uniqueID'].append(uniqids)
            user_datas[user]['pos'].append(poses)
            user_datas[user]['Morphological'].append(morphologicals)
            user_datas[user]['Dependency Label'].append(dependencyLabels)
            user_datas[user]['Dependency head'].append(dependencyHeads)
            user_datas[user]['gold'].append(golds)
            sentence_property = {}

    user_datas = dict(user_datas)
    for key, value in user_datas.items():
        user_datas[key] = dict(user_datas[key])
        _, perm = torch.sort(torch.Tensor(user_datas[key]['days']))
        for k, v in user_datas[key].items():
            user_datas[key][k] = [user_datas[key][k][p] for p in perm]

    for key, value in types.items():
        if key == 'vocab':
            continue
        else:
            types[key] = set(value)

    datas = feature_to_id(user_datas, model.dict, opt.gpu)
    for user, data in datas.items():
        user = model.dict['feature']['user'].labelToIdx[user]
        user_data = Variable(data['user'][0].unsqueeze(1)).cuda()
        for input, tag, days, session, format, time, pos, pre_ids in zip(data['token'], data['gold'], data['days'], data['session'], data['format'], data['time'], data['pos'], data['id']):
            #tag = Variable(tag.unsqueeze(1)).cuda()
            for l in sorted(model.user_history[user].items()):
                if l[0] <= days[0]:
                    near_day = l[0]
                else:
                    break
            input = Variable(input.unsqueeze(1)).cuda()
            pos = Variable(pos.unsqueeze(1)).cuda()
            days = Variable(days.unsqueeze(1)).cuda()
            session = Variable(session.unsqueeze(1)).cuda()
            format = Variable(format.unsqueeze(1)).cuda()
            time = Variable(time.unsqueeze(1)).cuda()
            history = Variable(model.user_history[user][near_day][0].unsqueeze(0)).cuda()
            history_hidden = (Variable(model.user_history[user][near_day][1].unsqueeze(1)).cuda(), Variable(model.user_history[user][near_day][2].unsqueeze(1)).cuda())
            pres = model.detect(input, tag, days, session, format, time, user_data, pos, history, history_hidden)
            for pre_id, pre in zip(pre_ids, pres):
                outF.write('{} {}\n'.format(pre_id, pre))

if __name__ == "__main__":
    main()
