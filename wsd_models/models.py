'''
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from wsd_models.util import *

def tokenize_glosses(gloss_arr, tokenizer, max_len):
    glosses = []
    masks = []
    for gloss_text in gloss_arr:
        g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token)+tokenizer.encode(gloss_text)+tokenizer.encode(tokenizer.sep_token)]
        g_attn_mask = [1]*len(g_ids)
        g_fake_mask = [-1]*len(g_ids)
        g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])
        g_ids = torch.cat(g_ids, dim=-1)
        g_attn_mask = torch.tensor(g_attn_mask)
        glosses.append(g_ids)
        masks.append(g_attn_mask)

    return glosses, masks

def mask_logits(target, mask, logit=-1e30):
    return target * mask + (1 - mask) * (logit)

def load_projection(path):
    proj_path = os.path.join(path, 'best_probe.ckpt')
    with open(proj_path, 'rb') as f: proj_layer = torch.load(f)
    return proj_layer

class PretrainedClassifier(torch.nn.Module):
    def __init__(self, num_labels, encoder_name, proj_ckpt_path):
        super(PretrainedClassifier, self).__init__()

        self.encoder, self.encoder_hdim = load_pretrained_model(encoder_name)

        if proj_ckpt_path and len(proj_ckpt_path) > 0:
            self.proj_layer = load_projection(proj_ckpt_path)
            #assert to make sure correct dims
            assert self.proj_layer.in_features == self.encoder_hdim
            assert self.proj_layer.out_features == num_labels
        else:
            self.proj_layer = torch.nn.Linear(self.encoder_hdim, num_labels)

    def forward(self, input_ids, input_mask, example_mask):
        output = self.encoder(input_ids, attention_mask=input_mask)[0]

        example_arr = []
        for i in range(output.size(0)):
            example_arr.append(process_encoder_outputs(output[i], example_mask[i], as_tensor=True))
        output = torch.cat(example_arr, dim=0)
        output = self.proj_layer(output)
        return output

class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss

    def forward(self, input_ids, attn_mask):
        #encode gloss text
        if self.is_frozen:
            with torch.no_grad():
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[-1][-4:]

        gloss_output = torch.cat([i.unsqueeze(0) for i in gloss_output], dim=0).mean(0)

        #training model to put all sense information on CLS token
        gloss_output = gloss_output[:,:,:].squeeze(dim=1)
        return gloss_output

class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context):
        super(ContextEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context

    def forward(self, input_ids, attn_mask, output_mask):
        #encode context
        if self.is_frozen:
            with torch.no_grad():
                context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[-1][-4:]
        context_output = torch.cat([i.unsqueeze(0) for i in context_output], dim=0).mean(0)
        #average representations over target word(s)
        example_arr = []
        for i in range(context_output.size(0)):
            example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)

        return context_output

class LinearAttention(nn.Module):
    def __init__(self, in_dim=300, mem_dim=300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(in_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(1e-2)
        self.linear1 = nn.Linear(in_dim, mem_dim)
        self.linear2 = nn.Linear(in_dim, mem_dim)
        torch.nn.init.xavier_normal_(self.linear.weight.data)
        torch.nn.init.xavier_normal_(self.linear1.weight.data)
        torch.nn.init.xavier_normal_(self.linear2.weight.data)

    def forward(self, feature, aspect_v, dmask, word='word'):
        Q = self.linear(aspect_v.float())
        Q = nn.functional.normalize(Q, dim=1)

        attention_s = torch.mm(Q, Q.T)
        attention_sk = mask_logits(attention_s, dmask, 0)

        if 'word' in word:
            new_feature = self.linear(feature.float())
            new_feature = nn.functional.normalize(new_feature, dim=2)

            feature_reshape = new_feature.reshape(new_feature.shape[0] * new_feature.shape[1], -1)
            attention_ww = torch.mm(feature_reshape, feature_reshape.T)
            attention_w = torch.stack(
                torch.stack(attention_ww.split(new_feature.shape[1]), dim=0).mean(1).squeeze(1).split(new_feature.shape[1],
                                                                                                      dim=1), dim=1).mean(2)
            attention_wk = mask_logits(attention_w, dmask, 0)

            att_weight = attention_sk + attention_wk
        else:
            att_weight = attention_sk

        att_weight[att_weight == 0] = -1e30
        attention = F.softmax(att_weight, dim=1)

        new_out = torch.mm(attention.half(), aspect_v)

        return new_out

class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, tie_encoders=False, num_heads=6):
        super(BiEncoderModel, self).__init__()

        #tying encoders for ablation
        self.tie_encoders = tie_encoders

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = ContextEncoder(encoder_name, freeze_context)
        if self.tie_encoders:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)
        else:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim
        self.gat = [LinearAttention(self.gloss_encoder.gloss_hdim, self.gloss_encoder.gloss_hdim).cuda() for _ in
                    range(num_heads)]

    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)

    def gloss_forward(self, gloss_input, gloss_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask)

    def gat_forward(self, gloss_input, gloss_mask, args, key_len_list, instances, pre_index, context_dict, senses=''):
        gloss_out_all = self.gloss_encoder.forward(gloss_input, gloss_mask)
        if 'sense' in args.gloss_mode:
            key_len = sum(key_len_list, [])
            adjacency_mat = torch.zeros(sum(key_len), sum(key_len))
            sense_index = [sum(key_len[:i]) for i in range(len(key_len))]
            if 'pred' in args.gloss_mode:
                p_index = [pre_index.get(inst, 0) for inst in instances]
                sense_index = [sense_index[i] + p_index[i] for i in range(len(p_index))]
            doc_sent = [('.'.join(i.split('.')[:-2]), int(i.split('.')[-2][1:]), '.'.join(i.split('.')[:-1])) for i in
                        instances]
            adjacency_mat[:, sense_index] = 1
            for i in range(len(instances)):
                index = []
                for s_index, sense in enumerate(sense_index):
                    if args.same:
                        if doc_sent[s_index][-1] not in context_dict[doc_sent[i][-1]]:
                            index.extend([i for i in range(sum(key_len[:s_index]), sum(key_len[:s_index + 1]))])
                    else:
                        if len(doc_sent[s_index][0]) > 2:
                            if doc_sent[s_index][0] != doc_sent[i][0] or abs(doc_sent[s_index][1] - doc_sent[i][1]) > 0:
                                index.extend([i for i in range(sum(key_len[:s_index]), sum(key_len[:s_index + 1]))])
                        elif abs(doc_sent[s_index][1] - doc_sent[i][1]) > 0:
                            index.extend([i for i in range(sum(key_len[:s_index]), sum(key_len[:s_index + 1]))])
                adjacency_mat[sum(key_len[:i]): sum(key_len[:i + 1]), index] = 0

            for k_index, j in enumerate(key_len):
                start, end = sum(key_len[:k_index]), sum(key_len[:k_index + 1])
                adjacency_mat[start: end, start: end] = 0

            adjacency_mat_f = adjacency_mat + torch.eye(sum(key_len))

            att_out = [att.forward(gloss_out_all[:, 1:-1, :], gloss_out_all[:, 0, :], adjacency_mat_f.cuda(),
                                   args.word).unsqueeze(1) for att in self.gat]
            att_out = torch.cat(att_out, dim=1)
            att_out = att_out.mean(dim=1)  # (N, D)min(31, gloss_out_all.shape[1]-1)
            assert len(gloss_out_all) == len(att_out)
            return att_out
        else:
            return gloss_out_all[:, 0, :]

#EOF