# referenced from https://github.com/ZexinYan/Medical-Report-Generation
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
sys.path.append("..")
from IUdata.build_vocab import Vocabulary, JsonReader
import numpy as np
from torchvision import transforms
import pickle


class ChestXrayDataSet(Dataset):
    def __init__(self,
                 image_dir,
                 data_json,
                 vocabulary,
                 imp_max,
                 fin_s_max,
                 fin_w_max,
                 transforms=None):
        self.image_dir = image_dir
        self.data = JsonReader(data_json)
        self.vocab = vocabulary
        self.transform = transforms
        # imp_max is the maximum number of words for impression
        self.imp_max = imp_max
        # fin_s_max is the maximum number of sentences in finding
        self.fin_s_max = fin_s_max
        # fin_w_max is the maximum number of words for one-sentence in finding
        self.fin_w_max = fin_w_max

    def __getitem__(self, index):
        frontal_img_name = self.data[index][0].split(',')[0]
        frontal_img = Image.open(os.path.join(self.image_dir, frontal_img_name + '.png')).convert('RGB')
        if self.transform is not None:
            frontal_img = self.transform(frontal_img)
        try:
            impression = self.data[index][1][0]
            finding = self.data[index][1][1]
        except Exception as err:
            impression = 'normal. '
            finding = 'normal. '

        # Define the impression target
        impression_tar = self._define_impression(impression)
        # Define the finding target
        finding_tar, finding_sen_num = self._define_finding(finding)

        # The allowed max length (set for maxpooling in sentence encoder)
        fix_max_imp_length = self.imp_max
        fix_fin_maxword_num = self.fin_w_max

        return frontal_img, frontal_img_name, impression_tar, \
               fix_max_imp_length, finding_tar, finding_sen_num, fix_fin_maxword_num

    def __len__(self):
        return len(self.data)

    def _define_impression(self, para):
        """ Define the impression"""
        imp = [self.vocab('<start>')]
        # Judge how many sentences are in the impression
        for i, sentence in enumerate(para.split('.')):
            # make words in sentence all the lowercase as well
            sentence = sentence.strip().lower().split()
            # allow one-word sentence in impression, since it will be combined into one sentence later
            if len(sentence) == 0:
                continue
            tokens = []
            # Since used split('.') in for loop, it will not have '.' in the sentence
            for token in sentence:
                # take "," as an individual punctuation
                if "," in token:
                    tokens.append(self.vocab(token[:-1]))
                    tokens.append(self.vocab(","))
                else:
                    tokens.append(self.vocab(token))
            # change the last ',' to '.' later
            tokens.append(self.vocab(','))
            imp.extend(tokens)
        imp.append(self.vocab('<end>'))
        # when the impression length is over max length, cut it to max length
        # and change the second last word to '.' and the last word to <end>
        if len(imp) > self.imp_max:
            imp = imp[:self.imp_max]
        imp[-2] = self.vocab('.')
        imp[-1] = self.vocab('<end>')
        return imp

    def _define_finding(self, para):
        """ Define the finding """
        fin = []
        for i, sentence in enumerate(para.split('.')):
            if i >= self.fin_s_max:
                break
            # make words in sentence all the lowercase as well
            sentence = sentence.strip().lower().split()
            # does not include the sentence with zero word, one word or over-max words for finding
            if len(sentence) == 0 or len(sentence) == 1:
                continue
            tokens = [self.vocab('<start>')]
            # Since used split('.') in for loop, it will not have '.' in the sentence
            for token in sentence:
                # take "," as an individual punctuation
                if "," in token:
                    tokens.append(self.vocab(token[:-1]))
                    tokens.append(self.vocab(","))
                else:
                    tokens.append(self.vocab(token))
            # add '.' as an individual punctuation at the end of sentence
            tokens.append(self.vocab('.'))
            tokens.append(self.vocab('<end>'))
            # when the finding length is over max length, cut it to max length and
            # change the second last word to '.' and the last word to <end>
            if len(tokens) > self.fin_w_max:
                tokens = tokens[:self.fin_w_max]
                tokens[-2] = self.vocab('.')
                tokens[-1] = self.vocab('<end>')
            fin.append(tokens)
        # # add a sentence with end sign to the end of finding
        empty_sen = [self.vocab('<end>')]
        # empty_sen = [self.vocab('<start>'), self.vocab('<end>')]
        fin.append(empty_sen)
        # The number of sentences in the finding
        sen_num = len(fin)
        return fin, sen_num


def _imp_targets(imps, max_imp_len):
    lengths = []
    # max_imp_len : (15,15,15,15...........15) The number of 15 is batch size
    targets = np.zeros((len(imps), max_imp_len[0]))
    for i, imp in enumerate(imps):
        targets[i, :len(imp)] = imp[:]
        lengths.append(len(imp))
    targets = torch.as_tensor(targets, dtype=torch.long)
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    return targets, lengths


def _fin_targets(fins, max_sen_num, fix_max_word_num):
    max_sen_num = max(max_sen_num)
    # fix_max_word_num : (15,15,15,15...........15) The number of 15 is batch size
    max_word_num = fix_max_word_num[0]
    lengths = []
    targets = np.zeros((len(fins), max_sen_num, max_word_num))
    # initial all sentences with end (empty sentence)
    targets[:, :, 0] = [2]
    for i, para in enumerate(fins):
        single_fin_len = []
        for j, sentence in enumerate(para):
            targets[i, j, :len(sentence)] = sentence[:]
            single_fin_len.append(len(sentence))
        lengths.append(single_fin_len)
    # set the length of all empty sentences with
    lengths = padding_lengths(lengths, max_sen_num)
    targets = torch.as_tensor(targets, dtype=torch.long)
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    return targets, lengths


def padding_lengths(lengths, max_sen_num):
    new_lengths = []
    for single_sentence in lengths:
        new_lengths.append(list(single_sentence + [1] * (max_sen_num - len(single_sentence))))
    return new_lengths


def collate_fn(data):
    frontal_img, f_img_id, impression_tar, fix_max_imp_length, finding_tar, finding_sen_num, fix_fin_maxword_num,  = zip(
        *data)

    frontal_images = torch.stack(frontal_img, 0)

    impression_targets, imp_lengths = _imp_targets(impression_tar, fix_max_imp_length)
    finding_targets, fin_lengths = _fin_targets(finding_tar, finding_sen_num, fix_fin_maxword_num)

    return frontal_images, f_img_id, impression_targets, finding_targets, imp_lengths, fin_lengths


def get_loader(image_dir, data_json, vocabulary,
               transform, batch_size, num_workers,
               imp_max, fin_s_max, fin_w_max, shuffle=False):
    dataset = ChestXrayDataSet(image_dir=image_dir,
                               data_json=data_json,
                               vocabulary=vocabulary,
                               imp_max=imp_max,
                               fin_s_max=fin_s_max,
                               fin_w_max=fin_w_max,
                               transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    vocab_path = '../IUdata/IUdata_vocab.pkl'
    image_dir = '../../datasets_origin/NLMCXR_Frontal'
    data_json = '../IUdata/IUdata_trainval.json'
    # file_list = '../data/new_data/debugging_data.txt'
    batch_size = 8
    resize = 256
    crop_size = 224
    num_workers = 2
    imp_max = 15
    fin_s_max = 7
    fin_w_max = 15

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(image_dir=image_dir,
                             data_json=data_json,
                             vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             imp_max=imp_max,
                             fin_s_max=fin_s_max,
                             fin_w_max = fin_w_max,
                             shuffle=True)
    # print the sample of one iteration
    for i, (frontal_imgs, f_id, impression_targets, finding_targets, imp_lengths, fin_lengths) in enumerate(
            data_loader):
        print('shape of frontal image', frontal_imgs.shape)
        print('frontal image id', f_id)
        print('impression target', impression_targets)
        print('shape of impression', impression_targets.shape)
        print('finding target', finding_targets)
        print('shape of finding', finding_targets.shape)
        print("impression lenghts", imp_lengths)
        print("finding lengths", fin_lengths)
        break
