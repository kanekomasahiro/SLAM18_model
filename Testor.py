import onmt
import torch.nn as nn
import torch
from torch.autograd import Variable
import BLSTM
#import Beam


class Testor(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage.cuda(opt.gpu))

        self.model_opt = checkpoint['opt']
        #print("batch_size", self.model_opt.batch_size)
        #self.vocab = checkpoint['dicts']['vocab']
        self.dict = checkpoint['dicts']
        self.user_history = checkpoint['user_history']

        #model = BLSTM.BLSTM(self.model_opt, self.vocab, checkpoint['type'], checkpoint['features'])
        model = BLSTM.BLSTM(self.model_opt, self.dict, checkpoint['type'], checkpoint['features'])

        model.load_state_dict(checkpoint['model'])

        if opt.cuda:
            model.cuda()
        else:
            model.cpu()

        self.model = model
        self.model.eval()


    def buildData(self, srcBatch, goldBatch):
        srcData = [self.all_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.all_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData,
            self.opt.batch_size, self.opt.cuda, volatile=True, sort_key=False)

    def buildTargetTokens(self, pred):
        tokens = self.all_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        """
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex.data[0]]
        """
        return tokens

    def translateBatch(self, batch):
        outputs = self.model(batch, test=True)
        #print(outputs.view(-1, batchSize, 3))
        #pre = torch.max(outputs, 1)[1]
        return outputs


    def detect(self, input, tag, days, session, format, time, user, pos, history, history_hidden):
        preds, _, _ = self.model(input, tag, user, days, session, format, time, pos, history, history_hidden, test=True)
        preds = [preds[i][1].data[0] for i in range(len(preds))]

        return preds
