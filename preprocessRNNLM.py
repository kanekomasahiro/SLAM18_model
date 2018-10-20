import onmt

import argparse
import torch
import nltk
from gensim.models import word2vec
from collections import defaultdict
import math
from gensim.models.wrappers.fasttext import FastText
from gensim.models import KeyedVectors

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_data', required=True,
                    help="Path to the training source data")
parser.add_argument('-valid_data', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_key', required=True,
                    help="Path to the validation source data")
parser.add_argument('-lang', default='en_w2v',
                    help="Language. [en_w2v|en_fast|es|fr]")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-vocab_size', type=int, default=40000,
                    help="Size of the source vocabulary")
parser.add_argument('-minimum_freq', type=int, default=0,
                    help="Size of the source vocabulary")

parser.add_argument('-batch_size', type=int, default=32,
                    help="Maximum batch size")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=1001,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-end_of_sentence', action='store_true', help='')

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def make_w2v(vocab):
    d = {}
    if opt.lang == 'en_w2v':
        model = KeyedVectors.load_word2vec_format('../../../GoogleNews-vectors-negative300.bin', binary=True)
    if opt.lang == 'en_fast':
        model = KeyedVectors.load_word2vec_format('../../../wiki-news-300d-1M.vec')
    if opt.lang == 'es':
        model = FastText.load_fasttext_format('../../../cc.es.300.bin')
    if opt.lang == 'fr':
        model = FastText.load_fasttext_format('../../../cc.fr.300.bin')
    for i in range(4, vocab.size()):
        word = vocab.idxToLabel[i]
        #if opt.lang == 'en_w2v':
            #if model.emb(word)[0] != None:
            #if model.emb(word)[0] != None:
                #d[i] = model.emb(word)
                #d[i] = model[word]
        if word in model:
            d[i] = model[word]
    return d


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


def makeFeature(data):
    vocabs = {}
    for key, value in data.items():
        in_d = {}
        in_d[onmt.Constants.UNK] = onmt.Constants.UNK_WORD
        in_d[onmt.Constants.PAD] = onmt.Constants.PAD_WORD
        vocab = onmt.Dict([value for key, value in sorted(in_d.items())], lower=opt.lower)
        for data in value:
            vocab.add(data)
        vocabs[key] = vocab
    return vocabs


def initFeature(name, data):

    # If a dictionary is still missing, generate it.
    print('Building ' + name + ' vocabulary...')
    vocab = makeFeature(data)

    print()
    return vocab


def makeVocabulary(data, size, freq):
    in_d = {}
    in_d[onmt.Constants.UNK] = onmt.Constants.UNK_WORD
    in_d[onmt.Constants.PAD] = onmt.Constants.PAD_WORD
    vocab = onmt.Dict([value for key, value in sorted(in_d.items())], lower=opt.lower)

    for key, value in data.items():
        vocab.add(key, value)

    originalSize = vocab.size()
    vocab = vocab.prune(size, freq)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, data, vocabSize, minimumFreq):

    # If a dictionary is still missing, generate it.
    print('Building ' + name + ' vocabulary...')
    genWordVocab = makeVocabulary(data, vocabSize, minimumFreq)

    vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def loadData(data, devKey=None):
    if devKey:
        keyD = {}
        keyF = open(devKey)
        for l in keyF:
            l = l.split()
            key_id = l[0]
            key_gold = int(l[1])
            keyD[key_id] = key_gold
    datas = []
    # ID化してモデルに入力する素性のタイプ数を把握するため
    types = {'user':list(), 'countries':list(), 'client':list(), 'session':list(), 'format':list(), 'uniqueID':list(), 'vocab':dict(), 'pos':list(), 'Morphological':list(), 'Dependency label':list()}
    for key in types.keys():
        if key != 'vocab':
            types[key] += [onmt.Constants.PAD_WORD]
            types[key] += [onmt.Constants.UNK_WORD]

    propety_size = {}
    dataF = open(data)
    #word_property = [[] for _ in range(7)]
    sentence_property = {}
    word_property = defaultdict(lambda: list())
    user_datas = defaultdict(lambda: defaultdict(lambda: list()))
    for d in dataF:
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
                        if devKey:
                            golds.append(keyD[sen_data])
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

    return user_datas, types


def feature_to_id(datas, dicts):
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
                        #else:
                            #new_datas[key][k] += [torch.LongTensor(dicts['feature'][k].convertToIdx(vv,
                                                    #onmt.Constants.UNK_WORD))]

    for key, value in datas.items():
        new_datas[key] = dict(new_datas[key])
        new_datas[key]['size'] = len(new_datas[key]['token'])

    return new_datas


def batching(datas):
    d = defaultdict(lambda: list())
    batchs = defaultdict(lambda: list())

    for data in datas:
        for key, value in data.items():
            if key != 'size':
                d[key] += [value]
    for key, value in d.items():
        max_problem = max([len(v) for v in value])
        for i in range(max_problem):
            problem_lengths = [v[i].size(0) for v in value if len(v) > i]
            max_length = max(problem_lengths)
            out = value[0][0].new(len(problem_lengths), max_length).fill_(onmt.Constants.PAD)
            for j, v in enumerate(value):
                if len(v) <= i:
                    break
                data_length = v[i].size(0)
                offset = max_length -data_length
                out[j].narrow(0, 0, data_length).copy_(v[i])
                out[j] = out[j].unsqueeze(0)
            batchs[key] += [out.t()]
    batchs = dict(batchs)
    return batchs


def data_to_batch(data, batch_size, shuffle=True):
    batchs = []
    data_size = len(data)

    print('... sorting users by size of sentences')
    perm = [d[0] for d in sorted(data.items(), key=lambda x: x[1]['size'], reverse=True)]

    batch_idx = 0

    for i in range(0, math.ceil(data_size/batch_size)):
        users = perm[i*batch_size:(i*batch_size)+batch_size]
        batchs += [batching([data[user] for user in users])]

    return batchs


def main():

    print('Making training data ...')
    train = {}
    train['data'], train['type'] = loadData(opt.train_data)
    feature_l = []
    keys = list(train['data'].keys())
    for key in sorted(train['data'][keys[0]].keys()):
        feature_l.append(key)
    feature_l.remove('gold')
    print('Making validation data ...')
    valid = {}
    valid['data'], valid['type'] = loadData(opt.valid_data, opt.valid_key)

    dicts = {}
    print('Making vocab ...')
    dicts['vocab'] = initVocabulary("vocab", train['type']['vocab'], opt.vocab_size, opt.minimum_freq)
    print('Making feature\'s vocab ...')
    dicts['feature'] = initFeature("feature", train['type'])

    print('Data to id')
    train['data'] = feature_to_id(train['data'], dicts)
    valid['data'] = feature_to_id(valid['data'], dicts)

    print('batch')
    train['data'] = data_to_batch(train['data'], opt.batch_size)
    valid['data'] = data_to_batch(valid['data'], opt.batch_size, shuffle=False)

    print('Extracting embeddings ...')
    dicts['w2v'] = make_w2v(dicts['vocab'])

    saveVocabulary("vocab", dicts["vocab"], opt.save_data + ".vocab.dict")

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'features': feature_l,
                 'train': train,
                 'valid': valid,
                 'opt': opt}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
