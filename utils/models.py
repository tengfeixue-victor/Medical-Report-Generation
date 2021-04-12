# Referenced from https://github.com/yunjey/pytorch-tutorial
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import init
import torch.nn.functional as F
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152, and extract local and global features"""
        super(EncoderCNN, self).__init__()

        # load pretrained model from ImageNet
        resnet = models.resnet152(pretrained=True)
        local_features_mod = list(resnet.children())[:8]
        global_features_mod = list(resnet.children())[8]

        self.resnet_local = nn.Sequential(*local_features_mod)
        self.resnet_global = nn.Sequential(global_features_mod)

    def forward(self, frontal_image):
        """Extract feature vectors from input images"""
        # Does not train convolutional layers
        with torch.no_grad():
            local_features = self.resnet_local(frontal_image)
            global_features = self.resnet_global(local_features).squeeze()

        return global_features


class Impression_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_global_features, num_conv1d_out=1024,
                 teach_forcing_rate=0.5, max_seq_length=15, dropout_rate=0):
        """Set the hyper-parameters and build the layers for impression decoder"""
        super(Impression_Decoder, self).__init__()
        # from frontal images
        self.visual_embed = nn.Linear(num_global_features, embed_size)
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentence_encoder = Sen_Encoder(embed_size, vocab_size, max_seq_length, num_conv1d_out)

        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.teach_forcing_rate = teach_forcing_rate
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_conv1d_out = num_conv1d_out

    def forward(self, global_features, impressions, imp_lengths):
        """Decode image feature vectors and generates the impression, and also global topic vector"""
        vis_embeddings = self.visual_embed(global_features)
        ini_input = vis_embeddings.unsqueeze(1)
        # impressions embedding
        imp_embedded = self.word_embed(impressions)
        decoder_input = torch.cat((ini_input, imp_embedded), 1)
        imp_packed = pack_padded_sequence(decoder_input, imp_lengths, batch_first=True, enforce_sorted=False)

        out_lstm, _ = self.lstm(imp_packed)
        padded_outs, _ = pad_packed_sequence(out_lstm, batch_first=True)
        decoder_outputs = F.log_softmax(self.dropout(self.linear(padded_outs)), dim=-1)
        decoder_outputs_packed = \
            pack_padded_sequence(decoder_outputs, imp_lengths, batch_first=True, enforce_sorted=False)[0]
        gt_packed = pack_padded_sequence(impressions, imp_lengths, batch_first=True, enforce_sorted=False)[0]
        _, predicted_sentences = decoder_outputs.max(dim=-1)
        if random.random() < self.teach_forcing_rate:
            topic_vector = self.sentence_encoder(impressions)
        else:
            topic_vector = self.sentence_encoder(predicted_sentences)

        return gt_packed, decoder_outputs_packed, topic_vector

    def sampler(self, global_features, max_len, ini_decoder_state=None):
        """"Generate the impression in the testing process"""
        vis_embeddings = self.visual_embed(global_features)
        ini_input = vis_embeddings.unsqueeze(1)
        decoder_input_t = ini_input
        decoder_state_t = ini_decoder_state
        impression_ids = []

        for i in range(max_len):
            decoder_output_t, decoder_state_t = self._forward_step(decoder_input_t, decoder_state_t)
            pre_values, pre_indices = decoder_output_t.max(dim=-1)
            pre_indices = pre_indices.unsqueeze(1)
            impression_ids.append(pre_indices)
            decoder_input_t = self.word_embed(pre_indices)

        impression_ids = torch.cat(impression_ids, dim=1)
        topic_vector = self.sentence_encoder(impression_ids)

        return impression_ids, topic_vector

    def _forward_step(self, input_t, state_t):
        """Used in testing process to generate word by word for impression"""
        output_t, state_t = self.lstm(input_t, state_t)
        out_t_squ = output_t.squeeze(dim=1)
        out_fc = F.log_softmax(self.linear(out_t_squ), dim=-1)
        return out_fc, state_t


class Sen_Encoder(nn.Module):
    def __init__(self, embed_size, vocab_size, seq_length, sen_enc_conv1d_out=1024):
        """Set the hyper-parameters and build the layers for sentence encoder."""
        super(Sen_Encoder, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.pool_size = seq_length
        self.conv1 = nn.Conv1d(embed_size, sen_enc_conv1d_out, kernel_size=3, stride=1)
        # Maxpooling is used to collapse the length of sentence to 1
        self.maxpool1 = nn.MaxPool1d(kernel_size=seq_length - 2)
        self.conv2 = nn.Conv1d(sen_enc_conv1d_out, sen_enc_conv1d_out, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=seq_length - 4)
        self.conv3 = nn.Conv1d(sen_enc_conv1d_out, sen_enc_conv1d_out, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=seq_length - 6)
        self.vocab_size = vocab_size

    def forward(self, pre_sentence):
        """"3Conv: Take impression or preceding sentence in finding
            and output the semantic feature of the sentence"""
        sen_embeddings = self.word_embed(pre_sentence)
        sen_embeddings_trans = sen_embeddings.transpose(1, 2)
        output1 = self.conv1(sen_embeddings_trans)
        out1_feature = self.maxpool1(output1).squeeze()
        output2 = self.conv2(output1)
        out2_feature = self.maxpool2(output2).squeeze()
        output3 = self.conv3(output2)
        out3_feature = self.maxpool3(output3).squeeze()
        sen_semantic = torch.cat((out1_feature, out2_feature, out3_feature), dim=-1)
        return sen_semantic


class Atten_Sen_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, decoder_num_layers, sen_enco_num_layers,
                 num_global_features, num_regions, num_conv1d_out=1024, teach_forcing_rate=1.0, max_seq_length=15,
                 max_sentence_num=7, dropout_rate=0):
        """Set the hyper-parameters and build the layers for attention decoder"""
        super(Atten_Sen_Decoder, self).__init__()
        self.embed_h = nn.Linear(num_global_features + num_conv1d_out * sen_enco_num_layers,
                                 hidden_size * decoder_num_layers)
        self.embed_c = nn.Linear(num_global_features + num_conv1d_out * sen_enco_num_layers,
                                 hidden_size * decoder_num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, decoder_num_layers, batch_first=True, dropout=dropout_rate)
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentence_encoder = Sen_Encoder(embed_size, vocab_size, max_seq_length, num_conv1d_out)

        self.decoder_num_layers = decoder_num_layers
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length
        self.max_sentence_num = max_sentence_num
        self.vocab_size = vocab_size
        self.num_regions = num_regions
        self.num_conv1d_out = num_conv1d_out
        self.teach_forcing_rate = teach_forcing_rate

    def forward(self, global_features, topic_vector, findings, fin_lengths):
        """Generate findings"""
        gt_packed, decoder_outputs_packed = None, None
        last_input, last_state = self._combine_vis_text(global_features, topic_vector)
        # sentence recurrent loop
        for num_sen in range(findings.shape[1]):
            # impressions embedding
            fin_sen_embedded = self.word_embed(findings[:, num_sen, :])
            decoder_input = torch.cat((last_input, fin_sen_embedded), dim=1)
            # The num of words for each sentence in the finding
            fin_sen_lengths = fin_lengths[:, num_sen]

            # packed ground truth for calculating the loss in trainer.py
            gt_fin_sen_packed = \
                pack_padded_sequence(findings[:, num_sen, :], fin_sen_lengths, batch_first=True, enforce_sorted=False)[
                    0]
            if num_sen == 0:
                gt_packed = gt_fin_sen_packed
            else:
                gt_packed = torch.cat((gt_packed, gt_fin_sen_packed), dim=0)

            # packed for lstm here
            fin_sen_packed = pack_padded_sequence(decoder_input, fin_sen_lengths, batch_first=True,
                                                  enforce_sorted=False)
            out_lstm, _ = self.lstm(fin_sen_packed, last_state)
            padded_outs, _ = pad_packed_sequence(out_lstm, batch_first=True)
            fin_sen_outputs = F.log_softmax(self.dropout(self.linear(padded_outs)), dim=-1)

            # packed for calculating the loss in trainer.py
            fin_sen_outputs_packed = \
                pack_padded_sequence(fin_sen_outputs, fin_sen_lengths, batch_first=True, enforce_sorted=False)[0]
            if num_sen == 0:
                decoder_outputs_packed = fin_sen_outputs_packed
            else:
                decoder_outputs_packed = torch.cat((decoder_outputs_packed, fin_sen_outputs_packed), dim=0)

            _, predicted_sentences = fin_sen_outputs.max(dim=-1)
            if random.random() < self.teach_forcing_rate:
                sen_vector = self.sentence_encoder(findings[:, num_sen, :])
            else:
                sen_vector = self.sentence_encoder(predicted_sentences)
            last_input, last_state = self._combine_vis_text(global_features, sen_vector)

        return gt_packed, decoder_outputs_packed

    def sampler(self, global_features, topic_vector, max_single_sen_len, max_sen_num, ini_decoder_state=None):
        """"Generate findings in the testing process"""

        last_input, last_state = self._combine_vis_text(global_features, topic_vector)
        # The dimension of predicted_findings is denoted at the end
        predicted_findings = []
        # sentence recurrent loop
        for num_sen in range(max_sen_num):
            # predicted sentences (indices), used for generating the preceding topic vector
            predicted_single_sentence = []

            # word recurrent loop
            for time_step in range(max_single_sen_len):
                decoder_output_t, decoder_state_t = self._word_step(last_input, last_state)
                last_state = decoder_state_t
                _, pre_indices = decoder_output_t.max(dim=-1)
                pre_indices = pre_indices.unsqueeze(1)
                predicted_single_sentence.append(pre_indices)
                last_input = self.word_embed(pre_indices)

            predicted_single_sentence = torch.cat(predicted_single_sentence, dim=1)
            sen_vector = self.sentence_encoder(predicted_single_sentence)
            last_input, last_state = self._combine_vis_text(global_features, sen_vector)
            predicted_single_sentence = predicted_single_sentence.unsqueeze(-1)
            predicted_findings.append(predicted_single_sentence)

        predicted_findings = torch.cat(predicted_findings, dim=2)
        predicted_findings = predicted_findings.transpose(1, 2)

        return predicted_findings

    def _combine_vis_text(self, global_features, sen_vec):
        """ Combine visual features with semantic sentence vector to get hidden and cell state"""
        ini_input = torch.zeros(global_features.shape[0]).long().to(device)
        last_input = self.word_embed(ini_input).unsqueeze(1)
        con_features = torch.cat((global_features, sen_vec), dim=1)
        h_stat = self.embed_h(con_features)
        h_stat = h_stat.view((h_stat.shape[0], self.decoder_num_layers, -1)).transpose(0, 1).contiguous()
        c_stat = self.embed_c(con_features)
        c_stat = c_stat.view((c_stat.shape[0], self.decoder_num_layers, -1)).transpose(0, 1).contiguous()
        last_state = (h_stat, c_stat)
        return last_input, last_state

    def _word_step(self, input_t, state_t):
        """generate sentence word by word"""
        output_t, state_t = self.lstm(input_t, state_t)
        out_t_squ = output_t.squeeze(dim=1)
        out_fc = F.log_softmax(self.linear(out_t_squ), dim=-1)
        return out_fc, state_t
