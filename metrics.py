import torch
import numpy as np
import pickle
import time
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _imp_generator(predicted_imps, batch_idx, vocab):
    """"generate the impression from predicted id"""
    impression_ids = predicted_imps[batch_idx]
    impression_words = []
    for word_id in impression_ids:
        word = vocab.id2word[word_id]
        if word == '<start>' or word == '<pad>' or word == "<unk>":
            continue
        # <end> will stop the generation of one sentence
        elif word == '<end>':
            break
        else:
            impression_words.append(word)
    impression = ' '.join(impression_words)
    # Take punctuation as a single word
    impression = impression.lower()
    return impression


def _fin_generator(predicted_fins, batch_idx, vocab):
    """"generate the finding from predicted id"""
    finding_ids = predicted_fins[batch_idx]
    # print(finding_ids)
    finding_sentences = []
    for num_sen in range(finding_ids.shape[0]):
        single_sentence_ids = finding_ids[num_sen]
        single_sentence_words = []
        for word_id in single_sentence_ids:
            word = vocab.id2word[word_id]
            if word == '<start>' or word == '<pad>' or word == "<unk>":
                continue
            # <end> will stop the generation of one sentence
            elif word == '<end>':
                break
            else:
                single_sentence_words.append(word)
        single_sentence = ' '.join(single_sentence_words)
        # empty sentence will stop the generation
        if not single_sentence:
            break
        else:
            # Take punctuation as a single word
            single_sentence = single_sentence.lower()
        finding_sentences.append(single_sentence)
    finding = ' '.join(finding_sentences)
    return finding


def _gt_imp_generator(para):
    """ Modify the ground truth impression to be consistent with our training.
        Every impression sentence must end with '.',
        all '.' in the middle of impression will be changed to ','   """
    para = para.split('.')
    gt_imp = ''
    for i, sentence in enumerate(para):
        # remove the leading or trailing spaces
        sentence = sentence.strip()
        if len(sentence) > 0:
            sentence = sentence + ' , '
        gt_imp = gt_imp + sentence
    gt_imp_lst = gt_imp.split()
    # change the last ',' to '.'
    gt_imp_lst[-1] = '.'
    gt_imp = ' '.join(gt_imp_lst)
    # leave space between the original , and the word before it
    gt_imp = gt_imp.lower().replace(', ', ' , ')
    # fix the two-space issue caused by last operation
    gt_imp = gt_imp.replace('  , ', ' , ')
    return gt_imp


def _gt_fin_generator(para):
    """Modify the ground truth finding to be consistent with our training
        Every finding sentence must be ended with '.'   """
    gt_fin = para.lower().replace(', ', ' , ').replace('. ', ' . ')
    return gt_fin


def _generate_imp_fin_dict(predicted_imps_lst, predicted_fins_lst, image_ids_lst, args):
    pre_imp_dict = {}
    pre_fin_dict = {}
    pre_imp_fin_dict = {}
    gt_imp_dict = {}
    gt_fin_dict = {}
    gt_imp_fin_dict = {}

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.eval_json_dir) as f:
        data = json.load(f)

    # The length of predicted impression lst is the number of batches
    assert len(predicted_imps_lst) == len(predicted_fins_lst) == len(image_ids_lst)
    for idx in range(len(predicted_imps_lst)):
        if torch.cuda.is_available():
            predicted_imps = predicted_imps_lst[idx].cpu().data.numpy()
            predicted_fins = predicted_fins_lst[idx].cpu().data.numpy()
        else:
            predicted_imps = predicted_imps_lst[idx].data.numpy()
            predicted_fins = predicted_fins_lst[idx].data.numpy()
        image_ids = np.asarray(image_ids_lst[idx])
        # shape 0 is the number of samples in a batch
        assert predicted_imps.shape[0] == predicted_fins.shape[0] == image_ids.shape[0]
        for batch_idx in range(predicted_imps.shape[0]):
            img_id = image_ids[batch_idx]
            # Impressions:
            pre_imp = _imp_generator(predicted_imps, batch_idx, vocab)
            gt_imp = data[img_id][0]
            gt_imp = _gt_imp_generator(gt_imp)
            pre_imp_dict[img_id] = [pre_imp]
            gt_imp_dict[img_id] = [gt_imp]

            # Findings:
            pre_fin = _fin_generator(predicted_fins, batch_idx, vocab)
            gt_fin = data[img_id][1]
            gt_fin = _gt_fin_generator(gt_fin)
            pre_fin_dict[img_id] = [pre_fin]
            gt_fin_dict[img_id] = [gt_fin]
            # Impression+Finding
            pre_imp_fin = pre_imp + ' ' + pre_fin
            gt_imp_fin = gt_imp + ' ' + gt_fin
            pre_imp_fin_dict[img_id] = [pre_imp_fin]
            gt_imp_fin_dict[img_id] = [gt_imp_fin]

    return gt_imp_dict, pre_imp_dict, gt_fin_dict, pre_fin_dict, gt_imp_fin_dict, pre_imp_fin_dict


def _define_metrics(gts, res):
    bleu_scorer = Bleu(n=4)
    bleu, _ = bleu_scorer.compute_score(gts=gts, res=res)

    rouge_scorer = Rouge()
    rouge, _ = rouge_scorer.compute_score(gts=gts, res=res)

    cider_scorer = Cider()
    cider, _ = cider_scorer.compute_score(gts=gts, res=res)

    meteor_scorer = Meteor()
    meteor, _ = meteor_scorer.compute_score(gts=gts, res=res)

    for i in range(4):
        bleu[i] = round(bleu[i], 4)

    return bleu, round(meteor, 4), round(rouge, 4), round(cider, 4)


def compute_metrics(predicted_imps_lst, predicted_fins_lst, image_ids_lst, args):
    gt_imp_dic, pre_imp_dic, gt_fin_dic, pre_fin_dic, gt_imp_fin_dic, pre_imp_fin_dic = \
        _generate_imp_fin_dict(predicted_imps_lst, predicted_fins_lst, image_ids_lst, args)
    if args.imp_fin_only:
        imp_fin_bleu, imp_fin_meteor, imp_fin_rouge, imp_fin_cider = _define_metrics(gt_imp_fin_dic, pre_imp_fin_dic)
        print('Impression + Finding: bleu = %s, meteor = %s, rouge = %s, cider = %s' % (
            imp_fin_bleu, imp_fin_meteor, imp_fin_rouge, imp_fin_cider))

    else:
        imp_bleu, imp_meteor, imp_rouge, imp_cider = _define_metrics(gt_imp_dic, pre_imp_dic)
        fin_bleu, fin_meteor, fin_rouge, fin_cider = _define_metrics(gt_fin_dic, pre_fin_dic)
        imp_fin_bleu, imp_fin_meteor, imp_fin_rouge, imp_fin_cider = _define_metrics(gt_imp_fin_dic, pre_imp_fin_dic)
        print(
            'Impression: bleu = %s, meteor = %s, rouge = %s, cider = %s' % (imp_bleu, imp_meteor, imp_rouge, imp_cider))
        print('Finding: bleu = %s, meteor = %s, rouge = %s, cider = %s' % (fin_bleu, fin_meteor, fin_rouge, fin_cider))
        print('Impression + Finding: bleu = %s, meteor = %s, rouge = %s, cider = %s' % (
            imp_fin_bleu, imp_fin_meteor, imp_fin_rouge, imp_fin_cider))

    return imp_fin_bleu, imp_fin_meteor, imp_fin_rouge, imp_fin_cider


def _writer(txt, imp_dict, fin_dict):
    assert imp_dict.keys() == fin_dict.keys()
    for key in imp_dict.keys():
        txt.write(key)
        txt.write("\n")
        txt.write("Impression: ")
        txt.write(imp_dict[key][0])
        txt.write("\n")
        txt.write("Findings: ")
        txt.write(fin_dict[key][0])
        txt.write("\n")


def generate_text_file(predicted_imps_lst, predicted_fins_lst, image_ids_lst, num_run, args):
    gt_imp_dic, pre_imp_dic, gt_fin_dic, pre_fin_dic, gt_imp_fin_dic, pre_imp_fin_dic = \
        _generate_imp_fin_dict(predicted_imps_lst, predicted_fins_lst, image_ids_lst, args)
    if isinstance(num_run, str):
        gt_txt = open("results/{}_gt_results_{}.txt".format(num_run + "run", time.strftime('%Y-%m-%d-%H-%M')), "+w")
        pre_txt = open("results/{}_pre_results_{}.txt".format(num_run + "run", time.strftime('%Y-%m-%d-%H-%M')), "+w")
    else:
        gt_txt = open("results/{}_gt_results_{}.txt".format(str(num_run + 1) + "run", time.strftime('%Y-%m-%d-%H-%M')),
                      "+w")
        pre_txt = open(
            "results/{}_pre_results_{}.txt".format(str(num_run + 1) + "run", time.strftime('%Y-%m-%d-%H-%M')), "+w")
    _writer(gt_txt, gt_imp_dic, gt_fin_dic)
    _writer(pre_txt, pre_imp_dic, pre_fin_dic)
