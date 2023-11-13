import copy
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import layer_utils as Layer
from torch.nn import init

from data_utils import shared_utils


class Baseline(nn.Module):
    def __init__(self, config, model_parameters):
        super(Baseline, self).__init__()
        self.config = config

        self.encoder = Layer.BERTCell(config.path.bert_model_path)  # define encoder layer

        self.embedding_dropout = nn.Dropout(model_parameters['embed_dropout'])  # define dropout layer
        """During training, randomly zeroes some of the elements of the input tensor with probability p using samples 
        from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call."""

        # define hyper-parameters.
        # use for evaluation
        if "loss_factor" in model_parameters:
            self.gamma = model_parameters['loss_factor']
        else:
            self.gamma = 1

        # define mapping full-connect layer.
        self.W = nn.ModuleList()
        for i in range(4):
            """
            nn.Linear nhận vào 2 tham số 
            in_features : size of each input sample
            out_features : size of each output sample
            self.encoder.hidden_size = 768
            config.val.norm_id_map = {'O': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}
            """
            self.W.append(copy.deepcopy(nn.Linear(self.encoder.hidden_size, len(config.val.norm_id_map))))
            # self.W.append(copy.deepcopy(nn.Linear(config.hidden_size*2, len(config.val.norm_id_map))))

        # define multi-crf decode the sequence.
        # có 4 CRF đại diện cho 4 phần tử để decode
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(copy.deepcopy(Layer.CRFCell(len(config.val.norm_id_map), batch_first=True)))


    def forward(self, input_ids, attn_mask, comparative_label=None, elem_label=None,
                result_label=None):
        # get token embedding.

        """
        đưa vào model input_ids and attn_mask, model trả về một array tuple trong đó 2 phần tử đầu có ý nghĩa như ở dưới
        sequence_output : Sequence of hidden-states at the output of the last layer of the model.
        pooled_output : Last layer hidden-state of the first token of the sequence (classification token) after further processing
        """
        token_embedding = self.encoder(input_ids, attn_mask)

        batch_size, sequence_length, _ = token_embedding.size()

        final_embedding = self.embedding_dropout(token_embedding)

        # linear mapping.
        multi_sequence_prob = [self.W[index](final_embedding) for index in range(len(self.W))]


        # decode sequence label.
        elem_output = []
        for index in range(3):
            # None is in dev and test mode
            if elem_label is None:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, None))
            else:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, elem_label[:, index, :]))

        # result extract sequence label.
        result_output = self.decoder[3](multi_sequence_prob[3], attn_mask, result_label)

        if elem_label is None and result_label is None:
            sent_output = []
            new_elem_output = []
            elem_output = torch.cat(elem_output, dim=0).view(3, batch_size, sequence_length).permute(1, 0, 2)
            for i in range(len(attn_mask)):
                non_padding_len = torch.count_nonzero(attn_mask[i])
                null_result_output = torch.cat(
                    (torch.zeros(non_padding_len, dtype=torch.int8, device="cuda"),
                     torch.tensor([-1] * (len(attn_mask[i]) - non_padding_len)).to("cuda")))
                if torch.equal(result_output[i], null_result_output):
                    # label for CEE
                    non_padding_len_2 = torch.count_nonzero(attn_mask[i])
                    null_elem_output = torch.cat(
                        (torch.zeros(non_padding_len_2, dtype=torch.int8, device="cuda"),
                         torch.tensor([-1] * (len(attn_mask[i]) - non_padding_len_2)).to("cuda")))
                    new_elem_output.append(torch.stack([null_elem_output, null_elem_output, null_elem_output]))
                    # label for CSI
                    sent_output.append(0)
                else:
                    new_elem_output.append(elem_output[i])
                    sent_output.append(1)
            sent_output = torch.tensor(sent_output).to("cuda")

            elem_feature = multi_sequence_prob
            elem_feature = [elem_feature[index].unsqueeze(0) for index in range(len(elem_feature))]
            elem_feature = torch.cat(elem_feature, dim=0).permute(1, 0, 2, 3)

            new_elem_output = torch.stack(new_elem_output, dim=0)
            return token_embedding, elem_feature, new_elem_output, result_output, sent_output

        # calculate sent loss and crf loss.
        # sent_loss = F.cross_entropy(sent_class_prob, comparative_label.view(-1))
        crf_loss = sum(elem_output) + result_output

        return self.gamma * crf_loss


class LSTMModel(nn.Module):
    def __init__(self, config, model_parameters, vocab, weight=None):
        super(LSTMModel, self).__init__()
        self.config = config

        # input embedding.
        if weight is not None:
            self.input_embed = nn.Embedding(len(vocab) + 10, config.input_size).from_pretrained(weight)
        else:
            self.input_embed = nn.Embedding(len(vocab) + 10, config.input_size)

        self.encoder = Layer.LSTMCell(
            config.input_size, config.hidden_size, config.num_layers,
            config.device, batch_first=True, bidirectional=True
        )
        self.embedding_dropout = nn.Dropout(model_parameters['embed_dropout'])

        # define hyper-parameters.
        if "loss_factor" in model_parameters:
            self.gamma = model_parameters['loss_factor']
        else:
            self.gamma = 1

        # define sentence classification
        # self.sent_linear = nn.Linear(self.encoder.hidden_size * 2, 2)

        # define mapping full-connect layer.
        self.W = nn.ModuleList()
        for i in range(4):
            self.W.append(copy.deepcopy(nn.Linear(self.encoder.hidden_size * 2, len(config.val.norm_id_map))))

        # define multi-crf decode the sequence.
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(copy.deepcopy(Layer.CRFCell(len(config.val.norm_id_map), batch_first=True)))

    def forward(self, input_ids, attn_mask, comparative_label=None, elem_label=None, result_label=None):
        # get token embedding.

        input_embedding = self.input_embed(input_ids)
        token_embedding, pooled_output = self.encoder(input_embedding)

        batch_size, sequence_length, _ = token_embedding.size()

        final_embedding = self.embedding_dropout(token_embedding)
        # class_embedding = self.embedding_dropout(pooled_output)

        # linear mapping.
        multi_sequence_prob = [self.W[index](final_embedding) for index in range(len(self.W))]
        # sent_class_prob = self.sent_linear(class_embedding)

        # decode sequence label.
        elem_output = []
        for index in range(3):
            if elem_label is None:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, None))
            else:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, elem_label[:, index, :]))

        # result extract sequence label.
        result_output = self.decoder[3](multi_sequence_prob[3], attn_mask, result_label)

        if elem_label is None and result_label is None:
            # _, sent_output = torch.max(torch.softmax(sent_class_prob, dim=1), dim=1)
            sent_output = []
            for i in range(len(attn_mask)):
                non_padding_len = torch.count_nonzero(attn_mask[i])
                null_result_output = torch.cat(
                    (torch.zeros(non_padding_len, dtype=torch.int8, device="cuda"),
                     torch.tensor([-1] * (len(attn_mask[i]) - non_padding_len)).to("cuda")))
                if torch.equal(result_output[i], null_result_output):
                    sent_output.append(0)
                else:
                    sent_output.append(1)
            sent_output = torch.tensor(sent_output).to("cuda")

            elem_output = torch.cat(elem_output, dim=0).view(3, batch_size, sequence_length).permute(1, 0, 2)

            elem_feature = multi_sequence_prob
            elem_feature = [elem_feature[index].unsqueeze(0) for index in range(len(elem_feature))]
            elem_feature = torch.cat(elem_feature, dim=0).permute(1, 0, 2, 3)

            # elem_feature: [B, 3, N, feature_dim]
            # result_feature: [B, N, feature_dim]
            return token_embedding, elem_feature, elem_output, result_output, sent_output

        # calculate sent loss and crf loss.
        # sent_loss = F.cross_entropy(sent_class_prob, comparative_label.view(-1))
        crf_loss = sum(elem_output) + result_output
        return self.gamma * crf_loss


class LogisticClassifier(nn.Module):
    def __init__(self, config, feature_dim, class_num=2, dropout=0.1, weight=None):
        super(LogisticClassifier, self).__init__()
        self.config = config
        self.class_num = class_num

        self.usingPredicate = False
        self.addBegin = False
        self.feature_type = 0

        self.feature_dim = feature_dim
        if self.feature_type == 0:
            self.fc = nn.Linear(4 * (768 + 5), 768)
            # self.fc = nn.Linear(768 + 4 * (768+5), 768)
            self.fc_2 = nn.Linear(768, 2)
        else:
            self.fc = nn.Linear(5 * 768, 768)
            self.fc_2 = nn.Linear(768, 2)

        if self.usingPredicate and self.feature_type == 0:
            self.fc_3 = nn.Linear(768 + 5, 768)
            # self.fc_3 = nn.Linear(768 + 768 + 5,  768)
            self.fc_4 = nn.Linear(768, 9)
        elif self.usingPredicate and self.feature_type == 2:
            self.fc_3 = nn.Linear(768 + 768, 768)
            self.fc_4 = nn.Linear(768, 9)
        elif not self.usingPredicate and self.feature_type == 0:
            self.fc_3 = nn.Linear(4 * 773, 768)
            # self.fc_3 = nn.Linear(768 + 4 * 773,  768)
            self.fc_4 = nn.Linear(768, 9)
        elif not self.usingPredicate and self.feature_type == 2:
            self.fc_3 = nn.Linear(5*768, 768)
            self.fc_4 = nn.Linear(768, 9)
        self.weight = weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_representation, pair_label=None):
        if pair_label is not None:
            valid_label, polarity_label = pair_label
            valid_label = torch.tensor(valid_label).long().to(self.config.device)
            polarity_label = torch.tensor(polarity_label).long().to(self.config.device)

            valid_indices = ~torch.isnan(pair_representation)
            valid_indices = torch.nonzero(valid_indices.all(dim=1)).squeeze().cpu().numpy()
            pair_representation = pair_representation[valid_indices]

            if self.usingPredicate:
                polarity_representation = pair_representation[:, 2319:3092]
            else:
                polarity_representation = pair_representation
        else:

            if self.usingPredicate:
                polarity_representation = pair_representation[:, :, :, 2319:3092]
            else:
                polarity_representation = pair_representation

        embed_valid = self.fc(pair_representation.view(-1, 4 * (768 + 5)))
        embed_valid = self.dropout(embed_valid)
        predict_label = self.fc_2(embed_valid)

        # label
        if self.usingPredicate:
            embed_label = self.fc_3(polarity_representation.view(-1, 773))
        else:
            embed_label = self.fc_3(polarity_representation.view(-1, 4 * 773))

        embed_label = self.dropout(embed_label)
        predict_label_2 = self.fc_4(embed_label)

        # calculate loss.
        if pair_label is not None:
            valid_label = valid_label[valid_indices]
            polarity_label = polarity_label[valid_indices]

            self.weight = self.weight.to(self.config.device)
            objective_fn_1 = nn.CrossEntropyLoss(weight=self.weight)
            objective_fn_2 = nn.CrossEntropyLoss()

            loss_1 = objective_fn_1(predict_label, valid_label.view(-1))
            loss_2 = objective_fn_2(predict_label_2, polarity_label.view(-1))
            return loss_1 + loss_2
        else:
            return torch.max(F.softmax(predict_label, dim=-1), dim=-1)[-1], \
            torch.max(F.softmax(predict_label_2, dim=-1), dim=-1)[-1]
