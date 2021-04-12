# Referenced from https://github.com/yunjey/pytorch-tutorial
import argparse
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import os
import pickle
import copy

from utils.load_data import get_loader
from utils.models import EncoderCNN, Impression_Decoder, Atten_Sen_Decoder
from metrics import compute_metrics, generate_text_file
from utils.logger import create_logger
from IUdata.build_vocab import JsonReader, Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_net(num_run, logger, args):
    logger.info("Start the {}th run of the same model".format(num_run + 1))

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # create log path
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Image preprocessing, normalization for the pretrained resnet
    train_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    # Data loader
    data_loader = get_loader(args.image_dir, args.json_dir,
                             vocab, train_transform, args.batch_size,
                             args.num_workers, args.max_impression_len,
                             args.max_sen_num, args.max_single_sen_len,
                             shuffle=True)
    # Models
    image_encoder = EncoderCNN().train().to(device)
    impression_decoder = Impression_Decoder(args.embed_size, args.hidden_size,
                                            vocab_size, args.imp_layers_num,
                                            args.num_global_features, args.num_conv1d_out, args.teach_rate,
                                            args.max_impression_len, dropout_rate=args.dropout_rate,
                                            ).train().to(device)
    finding_decoder = Atten_Sen_Decoder(args.embed_size, args.hidden_size, vocab_size,
                                        args.fin_num_layers, args.sen_enco_num_layers,
                                        args.num_global_features, args.num_regions, args.num_conv1d_out, args.teach_rate,
                                        args.max_single_sen_len, args.max_sen_num, dropout_rate=args.dropout_rate).train().to(device)

    # Initialize the best weights
    best_img_encoder = copy.deepcopy(image_encoder.state_dict())
    best_imp_decoder = copy.deepcopy(impression_decoder.state_dict())
    best_fin_decoder = copy.deepcopy(finding_decoder.state_dict())
    best_epoch = 0

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # no fc layers
    params_imp = list(impression_decoder.parameters())
    params_fin = list(finding_decoder.parameters())

    optimizer_imp = torch.optim.Adam(params_imp, lr=args.learning_rate)
    optimizer_fin = torch.optim.Adam(params_fin, lr=args.learning_rate)
    # Decay LR by a factor of 0.1 every 10 epochs
    scheduler_imp = torch.optim.lr_scheduler.StepLR(optimizer_imp, step_size=args.sche_step_size, gamma=args.sche_decay)
    scheduler_fin = torch.optim.lr_scheduler.StepLR(optimizer_fin, step_size=args.sche_step_size, gamma=args.sche_decay)

    # training process
    # initialize all metric values to zero
    imp_fin_bleu = [0, 0, 0, 0]
    imp_fin_meteor, imp_fin_rouge, imp_fin_cider = 0, 0, 0
    best_bleu = [0, 0, 0, 0]
    best_meteor, best_rouge, best_cider = 0, 0, 0
    pre_imps_lst, pre_fins_lst, img_id_lst = [], [], []
    best_pre_imps_lst, best_pre_fins_lst, best_img_id_lst = [], [], []
    # the value of bleu4 is used to if the model is improved
    imp_fin_bleu4_lst = [0]
    total_step = len(data_loader)
    for epoch in range(args.epochs):
        for i, (frontal_imgs, _, impressions, findings, imp_lengths, fin_lengths) in enumerate(data_loader):
            frontal_imgs = frontal_imgs.to(device)
            impressions = impressions.to(device)
            findings = findings.to(device)
            # impression
            global_feas = image_encoder(frontal_imgs)
            imp_targets, predicted_imp, global_topic_vec = impression_decoder(global_feas, impressions, imp_lengths)
            imp_loss = criterion(predicted_imp, imp_targets)
            fin_targets, predicted_fin = finding_decoder(global_feas, global_topic_vec, findings, fin_lengths)
            fin_loss = criterion(predicted_fin, fin_targets)

            # if args.train_separately:
            optimizer_imp.zero_grad()
            optimizer_fin.zero_grad()
            imp_loss.backward()
            fin_loss.backward()
            optimizer_imp.step()
            optimizer_fin.step()

            # Print log info
            if i % args.log_step == 0:
                # if args.train_separately:
                print('Epoch [{}/{}], Step [{}/{}], Imp Loss: {:.4f}, Imp Perplexity: {:5.4f}'
                      .format(epoch + 1, args.epochs, i, total_step, imp_loss.item(), np.exp(imp_loss.item())))
                print('Epoch [{}/{}], Step [{}/{}], Fin Loss: {:.4f}, Fin Perplexity: {:5.4f}'
                      .format(epoch + 1, args.epochs, i, total_step, fin_loss.item(), np.exp(fin_loss.item())))

        if (epoch + 1) % args.log_metrics_step == 0:
            pre_imps_lst, pre_fins_lst, img_id_lst = test_in_training(image_encoder, impression_decoder,
                                                                       finding_decoder, vocab, args)
            imp_fin_bleu, imp_fin_meteor, imp_fin_rouge, imp_fin_cider = compute_metrics(pre_imps_lst, pre_fins_lst,
                                                                                         img_id_lst, args)
            imp_fin_bleu4_lst.append(imp_fin_bleu[3])

        if max(imp_fin_bleu4_lst[:-1]) < imp_fin_bleu[3]:
            best_img_encoder = copy.deepcopy(image_encoder.state_dict())
            best_imp_decoder = copy.deepcopy(impression_decoder.state_dict())
            best_fin_decoder = copy.deepcopy(finding_decoder.state_dict())
            best_epoch = epoch
            best_bleu = imp_fin_bleu
            best_meteor = imp_fin_meteor
            best_rouge = imp_fin_rouge
            best_cider = imp_fin_cider
            best_pre_imps_lst = pre_imps_lst
            best_pre_fins_lst = pre_fins_lst
            best_img_id_lst = img_id_lst
            logger.info("The model performance improved when epoch [{}/{}], BLEU4:{}".format(epoch + 1, args.epochs,
                                                                                       imp_fin_bleu[3]))
        scheduler_imp.step()
        scheduler_fin.step()

    # save the best model
    torch.save(best_img_encoder, os.path.join(
        args.model_path, '{}-image_encoder-{}.ckpt'.format(num_run + 1, best_epoch + 1)))
    torch.save(best_imp_decoder, os.path.join(
        args.model_path, '{}-impression_decoder-{}.ckpt'.format(num_run + 1, best_epoch + 1)))
    torch.save(best_fin_decoder, os.path.join(
        args.model_path, '{}-finding_decoder-{}.ckpt'.format(num_run + 1, best_epoch + 1)))
    # generate the ground truth and predicted result
    generate_text_file(best_pre_imps_lst, best_pre_fins_lst, best_img_id_lst, num_run, args)
    logger.info(
        "Values of metric for the best model are: \n"
        " BLEU:{}, METEOR:{}, ROUGE:{}, Cider:{}".format(str(best_bleu),
                                                         str(best_meteor),
                                                         str(best_rouge),
                                                         str(best_cider)))
    logger.info("Save the best model weights when epoch [{}/{}], BLEU4:{}".format(best_epoch + 1,
                                                                                  args.epochs,
                                                                                  str(best_bleu[3])))
    logger.info('=' * 55)
    return best_bleu, best_meteor, best_rouge, best_cider


def test_in_training(image_encoder, impression_decoder, finding_decoder, vocab, args):
    # testing dataset loader
    test_transforms = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    eval_data_loader = get_loader(args.image_dir, args.eval_json_dir,
                                  vocab, test_transforms, args.eval_batch_size,
                                  args.num_workers, args.max_impression_len,
                                  args.max_sen_num, args.max_single_sen_len, shuffle=False)
    vocab_size = len(vocab)

    # Models
    image_encoder_eval = EncoderCNN().eval().to(device)
    impression_decoder_eval = Impression_Decoder(args.embed_size, args.hidden_size,
                                                 vocab_size, args.imp_layers_num,
                                                 args.num_global_features, args.num_conv1d_out,
                                                 args.teach_rate, args.max_impression_len).eval().to(device)
    finding_decoder_eval = Atten_Sen_Decoder(args.embed_size, args.hidden_size, vocab_size,
                                             args.fin_num_layers, args.sen_enco_num_layers,
                                             args.num_global_features, args.num_regions, args.num_conv1d_out,
                                             args.teach_rate, args.max_single_sen_len, args.max_sen_num).eval().to(device)

    image_encoder_eval.load_state_dict(image_encoder.state_dict())
    impression_decoder_eval.load_state_dict(impression_decoder.state_dict())
    finding_decoder_eval.load_state_dict(finding_decoder.state_dict())
    # Generate impressions and findings
    pre_imps_lst, pre_fins_lst, img_id_list = [], [], []
    for i, (images, images_ids, _, _, _, _) in enumerate(eval_data_loader):
        frontal_imgs = images.to(device)
        global_feas = image_encoder_eval(frontal_imgs)
        predicted_imps, global_topic_vec = impression_decoder_eval.sampler(global_feas, args.max_impression_len)
        predicted_fins = finding_decoder_eval.sampler(global_feas, global_topic_vec, args.max_single_sen_len,
                                                      args.max_sen_num)
        img_id_list.append(images_ids)
        pre_imps_lst.append(predicted_imps)
        pre_fins_lst.append(predicted_fins)

    return pre_imps_lst, pre_fins_lst, img_id_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--model_path', type=str, default='model_weights/', help='path for saving checkpoints')
    parser.add_argument('--vocab_path', type=str, default='IUdata/IUdata_vocab_0threshold.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='IUdata/NLMCXR_Frontal', help='directory for X-ray images')
    parser.add_argument('--json_dir', type=str, default='IUdata/IUdata_train.json', help='the path for json file')
    parser.add_argument('--log_path', default='./results', type=str, help='The path that stores the log files.')

    # model parameters
    parser.add_argument('--resize_size', type=int, default=256, help='The resize size of the X-ray image')
    parser.add_argument('--crop_size', type=int, default=224, help='The crop size of the X-ray image')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for loading data')
    parser.add_argument('--num_workers', type=int, default=4, help='multi-process data loading')
    parser.add_argument('--embed_size', type=int, default=512, help='The embed_size for vocabulary and images')
    parser.add_argument('--hidden_size', type=int, default=512, help='The number of hidden states in LSTM layers')
    parser.add_argument('--num_global_features', type=int, default=2048,
                        help='The number of global features for image encoder')
    parser.add_argument('--imp_layers_num', type=int, default=1, help='The number of LSTM layers in impression decoder')
    parser.add_argument('--fin_num_layers', type=int, default=2, help='The number of LSTM layers in finding decoder ')
    parser.add_argument('--sen_enco_num_layers', type=int, default=3,
                        help='The number of convolutional layer in topic encoder')
    parser.add_argument('--num_local_features', type=int, default=2048,
                        help='The channel number of local features for image encoder')
    parser.add_argument('--num_regions', type=int, default=49, help='The number of sub-regions for local features')
    parser.add_argument('--num_conv1d_out', type=int, default=1024, help='The number of output channels for 1d convolution of sentence encoder')

    # training parameters
    parser.add_argument('--teach_rate', type=float, default=1.0, help='The teach forcing rate in training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate in training')
    parser.add_argument('--epochs', type=int, default=50, help='The epochs in training')
    parser.add_argument('--sche_step_size', type=int, default=5,
                        help='The number of epochs for decay learning rate once')
    parser.add_argument('--sche_decay', type=float, default=0.9, help='The decay rate for learning rate')
    parser.add_argument('--log_step', type=int, default=50, help='The interval of displaying the loss and perplexity')
    parser.add_argument('--save_step', type=int, default=10, help='The interval of saving weights of models')
    parser.add_argument('--lambda_imp', type=float, default=0.5, help='The weight value for impression loss')
    parser.add_argument('--lambda_fin', type=float, default=0.5, help='The weight value for finding loss')
    parser.add_argument('--fix_image_encoder', type=bool, default=True, help='fix the image encoder or fine-tune it')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='The dropout rate for both encoder and decoder')
    parser.add_argument('--log_metrics_step', type=int, default=1, help='The interval of calculate the metrics')
    parser.add_argument('--train_separately', type=bool, default=True,
                        help="Train impression decoder and finding decoder separately or jointly")
    parser.add_argument('--imp_fin_only', type=bool, default=True, help='Only evaluate on Impression+Finding')

    # Evaluation parameters
    parser.add_argument('--eval_json_dir', type=str, default='IUdata/IUdata_test.json', help='the path for json file')
    parser.add_argument('--eval_batch_size', type=int, default=75, help='batch size for loading data')
    parser.add_argument('--max_impression_len', type=int, default=15,
                        help='The maximum length of the impression (one or several sentences)')
    parser.add_argument('--max_single_sen_len', type=int, default=15,
                        help='The maximum length of the each sentence in the finding')
    parser.add_argument('--max_sen_num', type=int, default=7, help='The maximum number of sentences in the finding')
    parser.add_argument('--single_punc', type=bool, default=True,
                        help='Take punctuation as a single word: If true, generate sentences such as: Hello , world .')

    args = parser.parse_args()
    print(args)
    # Record the training process and values
    logger = create_logger(args.log_path)
    logger.info('=' * 55)
    logger.info(args)
    logger.info('=' * 55)
    # The time of training the same model to get average results
    num_run = 3
    best_bleu_lst = []
    best_meteor_lst = []
    best_rouge_lst = []
    best_cider_lst = []
    for n_run in range(num_run):
        b_bleu, b_meteor, b_rouge, b_cider = train_net(n_run, logger, args)
        best_bleu_lst.append(b_bleu)
        best_meteor_lst.append(b_meteor)
        best_rouge_lst.append(b_rouge)
        best_cider_lst.append(b_cider)
    best_bleu_array = np.asarray(best_bleu_lst)
    avg_b_bleu = np.mean(best_bleu_array, axis=0)
    avg_b_meteor = sum(best_meteor_lst) / len(best_meteor_lst)
    avg_b_rouge = sum(best_rouge_lst) / len(best_rouge_lst)
    avg_b_cider = sum(best_cider_lst) / len(best_cider_lst)
    logger.info('*' * 55)
    logger.info(
        'Final Result -- Average values of metric for the best model are: \n'
        ' BLEU:{}, METEOR:{}, ROUGE:{}, Cider:{}'.format(
            str(avg_b_bleu),
            str(avg_b_meteor),
            str(avg_b_rouge),
            str(avg_b_cider)))