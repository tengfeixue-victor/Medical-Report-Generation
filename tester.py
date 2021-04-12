import argparse
import torch
import torchvision.transforms as transforms
import pickle

from utils.load_data import get_loader
from utils.models import EncoderCNN, Impression_Decoder, Atten_Sen_Decoder
from metrics import compute_metrics, generate_text_file
from IUdata.build_vocab import JsonReader, Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os


def test(img_en_path, imp_de_path, fin_de_path, args):
    """"load trained models and generate impressions and findings for evaluating"""
    test_transforms = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    # testing dataset loader
    eval_data_loader = get_loader(args.image_dir, args.eval_json_dir,
                                  vocab, test_transforms, args.eval_batch_size,
                                  args.num_workers, args.max_impression_len,
                                  args.max_sen_num, args.max_single_sen_len, shuffle=False)

    # Models
    image_encoder = EncoderCNN().eval().to(device)
    impression_decoder = Impression_Decoder(args.embed_size, args.hidden_size,
                                                 vocab_size, args.imp_layers_num,
                                                 args.num_global_features, args.num_conv1d_out,
                                                 args.teach_rate, args.max_impression_len).eval().to(device)
    finding_decoder = Atten_Sen_Decoder(args.embed_size, args.hidden_size, vocab_size,
                                             args.fin_num_layers, args.sen_enco_num_layers,
                                             args.num_global_features, args.num_regions, args.num_conv1d_out,
                                             args.teach_rate, args.max_single_sen_len, args.max_sen_num).eval().to(device)
    # load trained model weights
    image_encoder.load_state_dict(torch.load(img_en_path))
    impression_decoder.load_state_dict(torch.load(imp_de_path))
    finding_decoder.load_state_dict(torch.load(fin_de_path))

    # Generate impressions and findings
    pre_imps_lst, pre_fins_lst, img_id_list = [], [], []
    for i, (images, images_ids, _, _, _, _) in enumerate(eval_data_loader):
        frontal_imgs = images.to(device)
        global_feas = image_encoder(frontal_imgs)
        predicted_imps, global_topic_vec = impression_decoder.sampler(global_feas, args.max_impression_len)
        predicted_fins = finding_decoder.sampler(global_feas, global_topic_vec, args.max_single_sen_len,
                                                 args.max_sen_num)

        img_id_list.append(images_ids)
        pre_imps_lst.append(predicted_imps)
        pre_fins_lst.append(predicted_fins)

    return pre_imps_lst, pre_fins_lst, img_id_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--model_path', type=str, default='model_weights', help='path for weights')
    parser.add_argument('--vocab_path', type=str, default='IUdata/IUdata_vocab_0threshold.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='IUdata/NLMCXR_Frontal', help='directory for X-ray images')
    parser.add_argument('--eval_json_dir', type=str, default='IUdata/IUdata_test.json', help='the path for json file')
    # model parameters
    parser.add_argument('--eval_batch_size', type=int, default=75, help='batch size for loading data')
    parser.add_argument('--num_workers', type=int, default=2, help='multi-process data loading')
    parser.add_argument('--max_impression_len', type=int, default=15,
                        help='The maximum length of the impression (one or several sentences)')
    parser.add_argument('--max_single_sen_len', type=int, default=15,
                        help='The maximum length of the each sentence in the finding')
    parser.add_argument('--max_sen_num', type=int, default=7, help='The maximum number of sentences in the finding')
    parser.add_argument('--single_punc', type=bool, default=True,
                        help='Take punctuation as a single word: If true, generate sentences such as: Hello , world .')
    parser.add_argument('--imp_fin_only', type=bool, default=False, help='Only evaluate on Impression+Finding')

    #################################################################################################################################
    #################################################################################################################################
    # not changed parameters
    parser.add_argument('--resize_size', type=int, default=256, help='The resize size of the X-ray image')
    parser.add_argument('--crop_size', type=int, default=224, help='The crop size of the X-ray image')
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
    parser.add_argument('--num_conv1d_out', type=int, default=1024,
                        help='The number of output channels for 1d convolution of sentence encoder')
    parser.add_argument('--teach_rate', type=float, default=0.0, help='No teach force is used in testing')
    parser.add_argument('--log_step', type=int, default=100, help='The interval of displaying the loss and perplexity')
    parser.add_argument('--save_step', type=int, default=1000, help='The interval of saving weights of models')
    #################################################################################################################################
    #################################################################################################################################

    args = parser.parse_args()
    print(args)
    img_en_path, imp_de_path, fin_de_path = None, None, None
    num_ckpt = 0
    for path in os.listdir(args.model_path):
        if path.split(".")[-1] == 'ckpt':
            num_ckpt += 1
            if 'image' in path:
                img_en_path = os.path.join(args.model_path, path)
            elif 'impression' in path:
                imp_de_path = os.path.join(args.model_path, path)
            elif 'finding' in path:
                fin_de_path = os.path.join(args.model_path, path)

    """"Please only keep one combination of weights in the model_weights folder"""
    assert num_ckpt == 2 or num_ckpt == 3

    num_run = "test"
    predicted_imps_lst, predicted_fins_lst, image_id_list = test(img_en_path, imp_de_path, fin_de_path, args)
    _, _, _, _ = compute_metrics(predicted_imps_lst, predicted_fins_lst, image_id_list, args)
    generate_text_file(predicted_imps_lst, predicted_fins_lst, image_id_list, num_run, args)
