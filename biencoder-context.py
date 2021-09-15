'''
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
import os
import sys
import argparse
from tqdm import tqdm
import pickle
from pytorch_transformers import *
from collections import defaultdict
import random
import numpy as np

from wsd_models.util import *
from wsd_models.models import BiEncoderModel
from apex import amp
from copy import deepcopy

parser = argparse.ArgumentParser(description='Gloss Informed Bi-encoder for WSD')

#training arguments
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--weight', type=float, default=0.1, choices=[0.1, 0.2, 0.3, 0.4])
parser.add_argument('--opt_level', type=str, default='O2')
parser.add_argument('--train_mode', type=str, default='mean')
parser.add_argument('--train_data', type=str, default='semcor', choices=['semcor', 'semcor-wngt'])
parser.add_argument('--context_len', type=int, default=2, choices=[1, 2, 3, 4])
parser.add_argument('--current_epoch', type=int, default=0)
parser.add_argument('--context_mode', type=str, default='all', choices=['nonselect', 'nonwindow', 'all'])
parser.add_argument('--word', type=str, default='word', choices=['word', 'non'])
parser.add_argument('--same', action='store_true', help='whether the gloss and context encoder use the same context')
parser.add_argument('--gloss_mode', type=str, default='sense-pred', choices=['non', 'sense', 'sense-pred'])
parser.add_argument('--num_head', type=int, default=6)
parser.add_argument('--train_sent', type=int, default=1000000)
parser.add_argument('--dev_sent', type=int, default=1000000)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--silent', action='store_true', help='Flag to supress training progress bar for each epoch')
parser.add_argument('--lr', type=float, default=0.00001, choices=[1e-5, 5e-5, 1e-6, 5e-6])
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--context_max_length', type=int, default=128)
parser.add_argument('--gloss_max_length', type=int, default=32)
parser.add_argument('--step_mul', type=int, default=50, help='to slow down learning rate dropping after warmed up')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--context-bsz', type=int, default=4)
parser.add_argument('--gloss-bsz', type=int, default=400, choices=[150, 400])
parser.add_argument('--encoder-name', type=str, default='roberta-base',
    choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large', 'xlmroberta-base', 'xlmroberta-large'])
parser.add_argument('--ckpt', type=str, default='./data',
    help='filepath at which to save best probing model (on dev set)')
parser.add_argument('--data-path', type=str, default='./data/WSD_Evaluation_Framework',
    help='Location of top-level directory for the Unified WSD Framework')

#sets which parts of the model to freeze during training for ablation
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--sec_wsd', action='store_true')
parser.add_argument('--freeze_gloss', action='store_true')
parser.add_argument('--freeze_context', action='store_true')
parser.add_argument('--tie_encoders', action='store_true')

#evaluation arguments
parser.add_argument('--eval', action='store_true',
    help='Flag to set script to evaluate probe (rather than train)')
parser.add_argument('--split', type=str, default='semeval2007',
    choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL'],
    help='Which evaluation split on which to evaluate probe')


def tokenize_glosses(gloss_arr, tokenizer, max_len):
    glosses = []
    masks = []
    for gloss_text in gloss_arr:
        if 'xlm' in args.encoder_name:
            g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(gloss_text)]
        else:
            g_ids = [torch.tensor([[x]]) for x in
                 tokenizer.encode(tokenizer.cls_token) + tokenizer.encode(gloss_text) + tokenizer.encode(
                     tokenizer.sep_token)]
        g_attn_mask = [1]*len(g_ids)
        g_fake_mask = [-1]*len(g_ids)
        if 'xlm' in args.encoder_name:
            g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len,
                                                     pad_id=tokenizer.encode(tokenizer.pad_token)[1])
        else:
            g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len,
                                                 pad_id=tokenizer.encode(tokenizer.pad_token)[0])
        g_ids = torch.cat(g_ids, dim=-1)
        g_attn_mask = torch.tensor(g_attn_mask)
        glosses.append(g_ids)
        masks.append(g_attn_mask)

    return glosses, masks

#creates a sense label/ gloss dictionary for training/using the gloss encoder
def load_bn_glosses(data, tokenizer, bn_senses, lang, max_len):
    sense_glosses = {}

    count = 0
    for sent in data:
        for _, lemma, pos, _, label in sent:
            if label != -1:
                key_in = '%s#%s' % (lemma, pos)
                key = generate_key(lemma, pos)
                if key not in sense_glosses:
                    # get all sensekeys for the lemma/pos pair
                    sensekey_arr = bn_senses[key_in]
                    gloss_arr = [wn._synset_from_pos_and_offset(s[-1], int(s[3:-1])).definition() + ' ' + '. '.join(
                        wn._synset_from_pos_and_offset(s[-1], int(s[3:-1])).examples()) for s in sensekey_arr]
                    # preprocess glosses into tensors
                    gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
                    gloss_ids = torch.cat(gloss_ids, dim=0)
                    gloss_masks = torch.stack(gloss_masks, dim=0)
                    sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr)

                # make sure that gold label is retrieved synset
                if label not in sense_glosses[key][2]:
                    count += 1
    print(count)
    return sense_glosses

#creates a sense label/ gloss dictionary for training/using the gloss encoder
def load_and_preprocess_glosses(data, tokenizer, wn_senses, max_len=-1):
    sense_glosses = {}

    for sent in data:
        for _, lemma, pos, _, label in sent:
            if label == -1:
                continue  # ignore unlabeled words
            else:
                key = generate_key(lemma, pos)
                if key not in sense_glosses:
                    # get all sensekeys for the lemma/pos pair
                    sensekey_arr = wn_senses[key]
                    if max_len <= 32:
                        gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]
                    else:
                        gloss_arr = [wn.lemma_from_key(s).synset().definition() + ' ' + '. '.join(
                         wn.lemma_from_key(s).synset().examples()) for s in sensekey_arr]

                    # preprocess glosses into tensors
                    gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
                    gloss_ids = torch.cat(gloss_ids, dim=0)
                    gloss_masks = torch.stack(gloss_masks, dim=0)
                    sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr)

                # make sure that gold label is retrieved synset
                assert label in sense_glosses[key][2]

    return sense_glosses

def preprocess_context(tokenizer, text_data, gloss_dict=None, bsz=1, max_len=-1):
    if max_len == -1: assert bsz==1 #otherwise need max_length for padding

    context_ids = []
    context_attn_masks = []

    example_keys = []

    context_output_masks = []
    instances = []
    labels = []

    #tensorize data
    # print(tokenizer.encode(tokenizer.cls_token), tokenizer.encode(tokenizer.sep_token))
    for sent in (text_data):
        #cls token aka sos token, returns a list with index
        if 'xlm' in args.encoder_name:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)[1:-1]])]
        else:
            c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])]
        o_masks = [-1]
        sent_insts = []
        sent_keys = []
        sent_labels = []

        #For each word in sentence...
        key_len = []
        for idx, (word, lemma, pos, inst, label) in enumerate(sent):
            #tensorize word for context ids
            if 'xlm' in args.encoder_name:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())[1:-1]]
            else:
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
            c_ids.extend(word_ids)

            #if word is labeled with WSD sense...
            if label != -1:
                #add word to bert output mask to be labeled
                o_masks.extend([idx]*len(word_ids))
                #track example instance id
                sent_insts.append(inst)
                #track example instance keys to get glosses
                ex_key = generate_key(lemma, pos)
                sent_keys.append(ex_key)
                key_len.append(len(gloss_dict[ex_key][2]))
                sent_labels.append(label)
            else:
                #mask out output of context encoder for WSD task (not labeled)
                o_masks.extend([-1]*len(word_ids))

            #break if we reach max len
            if max_len != -1 and len(c_ids) >= (max_len-1):
                break

        if 'xlm' in args.encoder_name:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)[1:-1]])) #aka eos token
        else:
            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)]))  # aka eos token
        c_attn_mask = [1]*len(c_ids)
        o_masks.append(-1)
        assert len(c_ids) == len(o_masks)

        #not including examples sentences with no annotated sense data
        if len(sent_insts) > 0:
            context_ids.append(c_ids)
            context_attn_masks.append(c_attn_mask)
            context_output_masks.append(o_masks)
            example_keys.append(sent_keys)
            instances.append(sent_insts)
            labels.append(sent_labels)

    #package data
    context_dict = dict()

    doc_id, doc_seg = [], []
    for index, x in enumerate(instances):
        inst = '.'.join(x[0].split('.')[:-2])
        if inst not in doc_id:
            doc_id.append(inst)
            doc_seg.append(index)
    doc_seg.append(len(instances))
    new_context, new_attn_mask, new_out_mask = [], [], []

    from sklearn.feature_extraction.text import TfidfVectorizer

    for seg_index, seg_id in enumerate((doc_seg[:-1])):
        ids_c = context_ids[seg_id: doc_seg[seg_index + 1]]
        attn_masks_c = context_attn_masks[seg_id: doc_seg[seg_index + 1]]
        output_masks_c = context_output_masks[seg_id: doc_seg[seg_index + 1]]
        example_keys_c = example_keys[seg_id: doc_seg[seg_index + 1]]
        instances_c = instances[seg_id: doc_seg[seg_index + 1]]
        valid_instance = [i for i in instances_c[0] if i != -1][0]
        sent_ids = ['.'.join(i[0].split('.')[:-1]) for i in instances_c]
        if len(valid_instance.split('.')[0]) > 2:
            # doc = [' '.join(examp) for examp in example_keys_c]
            doc = [' '.join([i.split('+')[0] for i in examp if i.split('+')[1] in 'nvar']) for examp in example_keys_c]
            vectorizer = TfidfVectorizer()
            doc_mat = vectorizer.fit_transform(doc).toarray()
            for sent_id, vec in enumerate(doc_mat):
                scores = doc_mat[:, doc_mat[sent_id].nonzero()[0]].sum(1)
                id_score = [j for j in
                            sorted(zip([i for i in range(len(doc_mat))], scores), key=lambda x: x[1], reverse=True) if
                            j[0] != sent_id][:args.context_len]

                selected = [i[0] for i in id_score]
                # args.context_len = 2
                window_id = [i for i in range(len(doc_mat))][
                            max(sent_id - args.context_len, 0):sent_id + args.context_len + 1]
                pure_neighbor = [i for i in window_id if i != sent_id]
                #
                if args.context_mode == 'all':
                    ids = sorted(set(selected + [sent_id] + pure_neighbor))
                elif args.context_mode == 'nonselect':
                    ids = sorted(set([sent_id] + pure_neighbor))
                elif args.context_mode == 'nonwindow':
                    ids = sorted(set(selected + [sent_id]))
                else:
                    ids = [sent_id]

                total_len = len(sum([ids_c[i]for i in ids], []))
                while total_len > 512:
                    distance_index = sorted([(abs(s_id-sent_id), s_id) for s_id in ids], reverse=True)
                    ids.remove(distance_index[0][1])
                    total_len = len(sum([ids_c[i] for i in ids], []))
                if args.context_len > 0:
                    new_context.append(sum([ids_c[i]for i in ids], []))
                    new_attn_mask.append(sum([attn_masks_c[i] for i in ids], []))
                    new_out_mask.append(
                        sum([[-1] * len(output_masks_c[i]) if i != sent_id else output_masks_c[i] for i in ids], []))
                    assert len(new_context[-1]) == len(new_attn_mask[-1]) == len(new_out_mask[-1])
                else:
                    new_context.append(ids_c[sent_id])
                    new_attn_mask.append(attn_masks_c[sent_id])
                    new_out_mask.append(output_masks_c[sent_id])
                context_dict[sent_ids[sent_id]] = [sent_ids[i] for i in ids]
        else:
            new_context.extend(ids_c)
            new_attn_mask.extend(attn_masks_c)
            new_out_mask.extend(output_masks_c)

            for sent_id in sent_ids:
                context_dict[sent_id] = [sent_id]

    assert len(context_ids) == len(new_context)

    data = [list(i) for i in
            list(zip(new_context, new_attn_mask, new_out_mask, example_keys, instances, labels))]

    # print('Batching data with gloss length = {}...'.format(args.gloss_bsz))
    batched_data = []
    sent_index, current_list = [0], []
    sent_senses = [sum([len(gloss_dict[ex_key][2]) for ex_key in sent[3]]) for sent in data]
    for index, i in enumerate(sent_senses):
        current_list.append(i)
        if sum(current_list) > args.gloss_bsz:
            sent_index.append(index)
            current_list = current_list[-1:]
    sent_index.append(len(sent_senses))

    for index, data_index in enumerate(sent_index[:-1]):
        b = data[data_index: sent_index[index + 1]]
        max_len_b = max([len(x[1]) for x in b])
        if args.context_len > 0:
            max_len = max(max_len_b, max_len)
        for b_index, sent in enumerate(b):
            if 'xlm' in args.encoder_name:
                b[b_index][0], b[b_index][1], b[b_index][2] = normalize_length(sent[0], sent[1], sent[2], max_len,
                                                                           tokenizer.encode(tokenizer.pad_token)[1])
            else:
                b[b_index][0], b[b_index][1], b[b_index][2] = normalize_length(sent[0], sent[1], sent[2], max_len,
                                                                           tokenizer.encode(tokenizer.pad_token)[0])

        context_ids = torch.cat([torch.cat(x, dim=-1) for x, _, _, _, _, _ in b], dim=0)[:, :max_len_b]
        context_attn_mask = torch.cat([torch.tensor(x).unsqueeze(dim=0) for _, x, _, _, _, _ in b], dim=0)[:,
                            :max_len_b]
        context_output_mask = torch.cat([torch.tensor(x).unsqueeze(dim=0) for _, _, x, _, _, _ in b], dim=0)[:,
                              :max_len_b]
        example_keys = []
        for _, _, _, x, _, _ in b: example_keys.extend(x)
        instances = []
        for _, _, _, _, x, _ in b: instances.extend(x)
        labels = []
        for _, _, _, _, _, x in b: labels.extend(x)
        batched_data.append(
            (context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels))
    return batched_data, context_dict

def _train(train_data, model, gloss_dict, optim, schedule, train_index, train_dict, key_mat):
    model.train()

    # train_data, context_dict = train_data
    train_data = enumerate(train_data)
    train_data = tqdm(list(train_data))
    model.zero_grad()
    loss = 0.
    gloss_sz = 0
    context_sz = 0

    all_instance, pre_instance, mfs_instance, last_instance = 0, 0, 0, 0
    # with torch.no_grad():
    count = 0

    for b_index, (context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels) in train_data:

        sent_id, sent_seg = [], []
        key_len_list = []
        for in_index, inst in enumerate(instances):
            s_id = '.'.join(inst.split('.')[:-1])
            if s_id not in sent_id:
                sent_id.append(s_id)
                sent_seg.append(in_index)
        sent_seg.append(len(instances))

        for seg_index, seg in enumerate(sent_seg[:-1]):
            key_len_list.append([len(gloss_dict[key][2]) for key in example_keys[seg:sent_seg[seg_index + 1]]])

        total_sense = sum(sum(key_len_list, []))
        if 'large' in args.encoder_name:
            if total_sense > args.gloss_bsz:
                count += 1
                continue

        #run example sentence(s) through context encoder
        context_ids = context_ids.cuda()
        context_attn_mask = context_attn_mask.cuda()
        context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

        max_len_gloss = max(
            sum([[torch.sum(mask_list).item() for mask_list in gloss_dict[key][1]] for key in example_keys],
                []))
        gloss_ids_all = torch.cat([gloss_dict[key][0][:, :max_len_gloss] for key in example_keys])
        gloss_attn_mask_all = torch.cat([gloss_dict[key][1][:, :max_len_gloss] for key in example_keys])

        gloss_ids = gloss_ids_all.cuda()
        gloss_attn_mask = gloss_attn_mask_all.cuda()

        gat_out_all = model.gat_forward(gloss_ids, gloss_attn_mask, args, key_len_list, instances, train_index, train_dict)

        for seg_index, seg in enumerate(sent_seg[:-1]):
            current_example_keys = example_keys[seg: sent_seg[seg_index + 1]]
            current_key_len = key_len_list[seg_index]
            current_context_output = context_output[seg: sent_seg[seg_index + 1], :]
            current_insts = instances[seg: sent_seg[seg_index + 1]]
            current_labels = labels[seg: sent_seg[seg_index + 1]]

            gat_out = gat_out_all[
                        sum(sum(key_len_list[:seg_index], [])): sum(sum(key_len_list[:seg_index + 1], [])),
                        :]

            c_senses = sum([gloss_dict[key][2] for key in current_example_keys], [])
            gat_cpu = gat_out.cpu()
            assert len(c_senses) == len(gat_out)
            for k_index, key in enumerate(c_senses):
                key_mat[key] = gat_cpu[k_index:k_index + 1]

            gloss_output_pad = torch.cat([F.pad(
                gat_out[sum(current_key_len[:i]): sum(current_key_len[:i+1]), :],
                pad=[0, 0, 0, max(current_key_len) - j]).unsqueeze(0) for i, j in enumerate(current_key_len)], dim=0)

            out = torch.bmm(gloss_output_pad, current_context_output.unsqueeze(2)).squeeze(2)
            gloss_sz += gat_out.size(0)
            context_sz += 1

            for j, (key, label) in enumerate(zip(current_example_keys, current_labels)):
                idx = gloss_dict[key][2].index(label)
                label_tensor = torch.tensor([idx]).cuda()
                train_index[current_insts[j]] = out[j:j + 1, :current_key_len[j]].argmax(dim=1).item()
                loss += F.cross_entropy(out[j:j + 1, :current_key_len[j]], label_tensor)
                all_instance += 1
                if out[j:j + 1, :current_key_len[j]].argmax(dim=1).item() == idx:
                    pre_instance += 1
                if idx == 0:
                    mfs_instance += 1

        loss = loss / gloss_sz
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

        optim.step()
        schedule.step() # Update learning rate schedule

        #reset loss and gloss_sz
        loss = 0.
        gloss_sz = 0

        #reset model
        model.zero_grad()
        torch.cuda.empty_cache()

    print(count)
    print(pre_instance / all_instance, mfs_instance / all_instance)

    return model, optim, schedule, key_mat

def _eval(eval_data, model, gloss_dict, dev_index, dev_dict, key_mat=None, eval_file='ALL'):
    model.eval()
    csi_data = pickle.load(open('./data/csi_data', 'rb'))
    tag_lemma, tag_sense = pickle.load(open('./data/tag_semcor.txt', 'rb'))
    zsl, zss = [], []
    eval_preds = []
    gold_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/{}.gold.key.txt'.format(eval_file, eval_file))
    gold_labels = {i.split()[0]: i.split()[1:] for i in open(gold_path, 'r').readlines()}
    name = locals()
    dataset_name = sorted(set([i.split('.')[0] for i in gold_labels]))
    pos_tran = {'a': 'ADJ', 'n': 'NOUN', 'r': 'ADV', 'v': 'VERB'}
    for i in dataset_name:
        name['pred_c_%s' % i], name['pred_all_%s' % i] = 0, 0
    for pos in pos_tran.values():
        name['pred_c_%s' % pos], name['pred_all_%s' % pos] = 0, 0
    mfs_list, lfs_list = [], []
    correct_id = []
    if key_mat:
        key_dict, vec_list = {}, []
        count = 0
        for key, vec in key_mat.items():
            if '%' in key:
                synset = wn.lemma_from_key(key).synset().name()
            else:
                synset = wn._synset_from_pos_and_offset(key[-1], int(key[3:-1])).name()
            if synset not in key_dict:
                key_dict[synset] = count
                vec_list.append([vec])
                count += 1
            else:
                vec_list[key_dict[synset]].append(vec)
        vec_list = [torch.mean(torch.cat(i), dim=0).unsqueeze(0) for i in vec_list]
        vec_mat = torch.cat(vec_list).cuda()

    related_synset = {}
    for context_ids, context_attn_mask, context_output_mask, example_keys, insts, _ in tqdm(eval_data):
        with torch.no_grad():

            context_ids = context_ids.cuda()
            context_attn_mask = context_attn_mask.cuda()
            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)
            max_len_gloss = max(
                sum([[torch.sum(mask_list).item() for mask_list in gloss_dict[key][1]] for key in example_keys],
                    []))
            gloss_ids_all = torch.cat([gloss_dict[key][0][:, :max_len_gloss] for key in example_keys])
            gloss_attn_mask_all = torch.cat([gloss_dict[key][1][:, :max_len_gloss] for key in example_keys])

            gloss_ids = gloss_ids_all.cuda()
            gloss_attn_mask = gloss_attn_mask_all.cuda()

            sent_id, sent_seg = [], []
            key_len_list = []
            for in_index, inst in enumerate(insts):
                s_id = '.'.join(inst.split('.')[:-1])
                if s_id not in sent_id:
                    sent_id.append(s_id)
                    sent_seg.append(in_index)
            sent_seg.append(len(insts))

            for seg_index, seg in enumerate(sent_seg[:-1]):
                key_len_list.append([len(gloss_dict[key][2]) for key in example_keys[seg:sent_seg[seg_index + 1]]])

            senses = [gloss_dict[key][2] for key in example_keys]
            gat_out_all = model.gat_forward(gloss_ids, gloss_attn_mask, args, key_len_list, insts, dev_index, dev_dict)

            for seg_index, seg in enumerate(sent_seg[:-1]):
                current_example_keys = example_keys[seg: sent_seg[seg_index + 1]]
                current_key_len = key_len_list[seg_index]
                current_context_output = context_output[seg: sent_seg[seg_index + 1], :]
                current_insts = insts[seg: sent_seg[seg_index + 1]]

                gat_out = gat_out_all[
                          sum(sum(key_len_list[:seg_index], [])): sum(sum(key_len_list[:seg_index + 1], [])),
                          :]

                if key_mat:
                    c_senses = sum([gloss_dict[key][2] for key in current_example_keys], [])
                    gat_cpu = gat_out.cpu()
                    assert len(c_senses) == len(gat_out)
                    for k_index, key in enumerate(c_senses):
                        key_mat[key] = gat_cpu[k_index:k_index+1]

                    context_key = torch.mm(current_context_output, vec_mat.T)

                gloss_output_pad = torch.cat([F.pad(
                    gat_out[sum(current_key_len[:i]): sum(current_key_len[:i + 1]), :],
                    pad=[0, 0, 0, max(current_key_len) - j]).unsqueeze(0) for i, j in enumerate(current_key_len)],
                                             dim=0)

                out = torch.bmm(gloss_output_pad, current_context_output.unsqueeze(2)).squeeze(2).float().cpu()
                for j, key in enumerate(current_example_keys):
                    pred_idx = out[j:j + 1, :current_key_len[j]].topk(1, dim=-1)[1].squeeze().item()
                    if args.sec_wsd and len(gloss_dict[key][2]) >= 2:
                        if '%' in gloss_dict[key][2][0]:
                            synsets = [wn.lemma_from_key(i).synset().name() for i in gloss_dict[key][2]]
                        else:
                            synsets = [wn._synset_from_pos_and_offset(s[-1], int(s[3:-1])).name() for s in
                                       gloss_dict[key][2]]
                        key_sim = [(index, (k, sim)) for index, (k, sim) in
                                   enumerate(zip(synsets, out[j, :current_key_len[j]].tolist()))]
                        key_sim = sorted(key_sim, key=lambda x: x[1][1], reverse=True)
                        sec_sim = [0] * len(synsets)
                        for n_index, (index, (synset, _)) in enumerate(key_sim):
                            if n_index <= 2:
                                if synset in csi_data[0]:
                                    sec_synsets = sum([csi_data[1][q] for q in csi_data[0][synset]], [])
                                else:
                                    sec_synsets = []
                                if sec_synsets:
                                    sim = np.mean(sorted(
                                        [context_key[j, key_dict[k]].item() for k in sec_synsets if
                                         k in key_dict], reverse=True)[:1])
                                else:
                                    sim = out[j, index]
                            else:
                                sim = 0
                            sec_sim[index] = sim
                        sec_sim = torch.tensor(sec_sim)
                        pred_idx = ((1 - args.weight) *out[j, :current_key_len[j]] +
                                    args.weight *sec_sim).topk(1, dim=-1)[1].squeeze().item()

                    # if current_insts[j] not in dev_index:
                    dev_index[current_insts[j]] = pred_idx
                    pred_label = gloss_dict[key][2][pred_idx]
                    eval_preds.append((current_insts[j], pred_label))
                    if key not in tag_lemma:
                        if len(gloss_dict[key][2]) > 1:
                            if pred_label in gold_labels[current_insts[j]]:
                                zsl.append(1)
                            else:
                                zsl.append(0)
                    if not tag_sense.intersection(gold_labels[current_insts[j]]):
                        if len(gloss_dict[key][2]) > 1:
                            if pred_label in gold_labels[current_insts[j]]:
                                zss.append(1)
                            else:
                                zss.append(0)
                    if set(gloss_dict[key][2][:1]).intersection(gold_labels[current_insts[j]]):
                        if pred_label in gold_labels[current_insts[j]]:
                            mfs_list.append(1)
                        else:
                            mfs_list.append(0)
                    else:
                        if pred_label in gold_labels[current_insts[j]]:
                            lfs_list.append(1)
                        else:
                            lfs_list.append(0)
                    for i in dataset_name:
                        if i in current_insts[j]:
                            name['pred_all_%s' % i] += 1
                            if pred_label in gold_labels[current_insts[j]]:
                                name['pred_c_%s' % i] += 1
                                correct_id.append(current_insts[j])
                    for pos in pos_tran.values():
                        if pos in pos_tran[key.split('+')[1]]:
                            name['pred_all_%s' % pos] += 1
                            if pred_label in gold_labels[current_insts[j]]:
                                name['pred_c_%s' % pos] += 1

    correct_pred, all_pred = 0, 0
    for i in dataset_name:
        correct_pred += name['pred_c_%s' % i]
        all_pred += name['pred_all_%s' % i]
        if '2007' in i:
            print(i, name['pred_c_%s' % i]/name['pred_all_%s' % i], end='\t')
    for pos in pos_tran.values():
        if name['pred_all_%s' % pos] != 0:
            print(pos, name['pred_c_%s' % pos] / name['pred_all_%s' % pos], end='\t')

    print('ALL', correct_pred/all_pred, all_pred)
    print('zss %d, zsl %d' % (len(zss), len(zsl)), 'zss %f, zsl %f' % (sum(zss) / len(zss), sum(zsl) / len(zsl)))
    print(sum(mfs_list) / len(mfs_list), sum(lfs_list) / len(lfs_list), len(mfs_list), len(lfs_list),
          len(mfs_list + lfs_list))

    return eval_preds, dev_index, key_mat

def train_model(args):
    print('Training WSD bi-encoder model...')
    # if args.freeze_gloss: assert args.gloss_bsz == -1 #no gloss bsz if not training gloss encoder, memory concerns

    #create passed in ckpt dir if doesn't exist
    if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

    '''
    LOAD PRETRAINED TOKENIZER, TRAIN AND DEV DATA
    '''
    print('Loading data + preprocessing...')
    sys.stdout.flush()

    tokenizer = load_tokenizer(args.encoder_name)

    #loading WSD (semcor) data
    train_path = os.path.join(args.data_path, 'Training_Corpora/SemCor/')
    train_data = load_data(train_path, args.train_data, args.train_sent)

    #filter train data for k-shot learning
    # if args.kshot > 0: train_data = filter_k_examples(train_data, args.kshot)

    #dev set = semeval2007
    eval_file = 'ALL'
    semeval2007_path = os.path.join(args.data_path, 'Evaluation_Datasets/%s/' % eval_file)
    all_data = load_data(semeval2007_path, eval_file)[:args.dev_sent]

    #load gloss dictionary (all senses from wordnet for each lemma/pos pair that occur in data)
    wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
    wn_senses = load_wn_senses(wn_path)
    train_gloss_dict = load_and_preprocess_glosses(train_data, tokenizer, wn_senses, max_len=args.gloss_max_length)
    all_gloss_dict = load_and_preprocess_glosses(all_data, tokenizer, wn_senses, max_len=args.gloss_max_length)

    #preprocess and batch data (context + glosses)
    train_data, train_dict = preprocess_context(tokenizer, train_data, train_gloss_dict, bsz=args.context_bsz,
                                                max_len=args.context_max_length)
    all_data, dev_dict = preprocess_context(tokenizer, all_data, all_gloss_dict, bsz=args.context_bsz,
                                            max_len=args.context_max_length)

    epochs = args.epochs
    overflow_steps = -1
    t_total = len(train_data)*args.step_mul
    update_count = 0

    # if few-shot training, override epochs to calculate num. epochs + steps for equal training signal
    # if args.kshot > 0:
        # hard-coded num. of steps of fair kshot evaluation against full model on default numer of epochs
        # NUM_STEPS = 181500 # num batches in full train data (9075) * 20 epochs
        # num_batches = len(train_data)
        # epochs = NUM_STEPS//num_batches # recalculate number of epochs
        # overflow_steps = NUM_STEPS%num_batches # num steps in last overflow epoch (if there is one, otherwise 0)
        # t_total = NUM_STEPS # manually set number of steps for lr schedule
        # if overflow_steps > 0: epochs+=1 # add extra epoch for overflow steps
        # print('Overriding args.epochs and training for {} epochs...'.format(epochs))

    '''
    SET UP FINETUNING MODEL, OPTIMIZER, AND LR SCHEDULE
    '''
    if not args.continue_train:
        model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context,
                               tie_encoders=args.tie_encoders, num_heads=args.num_head)
        model = model.cuda()
    else:
        model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context,
                               tie_encoders=args.tie_encoders, num_heads=args.num_head)
        model_path = os.path.join(args.ckpt, 'best_model_%s.ckpt' % args.train_mode)
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()

    #optimize + scheduler from pytorch_transformers package
    #this taken from pytorch_transformers finetuning code
    weight_decay = 0.0 #this could be a parameter
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]}
        ]
    adam_epsilon = 1e-8
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon, weight_decay=weight_decay)
    schedule = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup, t_total=t_total)

    '''
    TRAIN MODEL ON SEMCOR DATA
    '''

    best_dev_f1 = 0.
    print('Training probe...')
    sys.stdout.flush()

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    train_index = dict()
    dev_pred = dict()
    dev_pred_list = [{}]
    key_mat_list = []
    mdev_pred_list = defaultdict(list)
    mkey_mat_list = defaultdict(list)

    key_mat = dict()
    for epoch in range(1, epochs+1):

        args.current_epoch = epoch
        #if last epoch, pass in overflow steps to stop epoch early
        train_steps = -1
        if epoch == epochs and overflow_steps > 0: train_steps = overflow_steps

        #train model for one epoch or given number of training steps
        model, optimizer, schedule, key_mat = _train(train_data, model, train_gloss_dict, optimizer, schedule,
                                                        train_index, train_dict, key_mat)
        key_mat_copy = {i: j for i,j in key_mat.items()}
        key_mat_list.append(key_mat_copy)

        eval_preds, dev_pred, key_mat = _eval(all_data, model, all_gloss_dict, dev_pred, dev_dict, key_mat)

        dev_pred_list.append(deepcopy(dev_pred))

        if 'mul' in args.train_mode:
            mdev_pred_list = evaluate_model_mul(args, tokenizer, model, mdev_pred_list, key_mat)

        #generate predictions file
        pred_filepath = os.path.join(args.ckpt, 'tmp_predictions_%s.txt' % args.train_mode)
        with open(pred_filepath, 'w') as f:
            for inst, prediction in eval_preds:
                if '2007' in inst:
                    f.write('{} {}\n'.format('.'.join(inst.split('.')[1:]), prediction))

        #run predictions through scorer
        eval_file = args.split
        gold_filepath = os.path.join(args.data_path, 'Evaluation_Datasets/%s/%s.gold.key.txt' % (eval_file, eval_file))
        scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
        _, _, dev_f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
        open('./data/dev_result.txt', 'a+').write('%s-%d-%f\n' % (args.train_mode, epoch, dev_f1))
        print('Dev f1 after {} epochs = {}'.format(epoch, dev_f1))
        sys.stdout.flush()

        if dev_f1 >= best_dev_f1:
            print('updating best model at epoch {}...'.format(epoch))
            sys.stdout.flush()
            best_dev_f1 = dev_f1
            #save to file if best probe so far on dev set
            model_fname = os.path.join(args.ckpt, 'best_model_%s.ckpt' % args.train_mode)
            with open(model_fname, 'wb') as f:
                torch.save(model.state_dict(), f)
            sys.stdout.flush()
            pickle.dump(dev_pred_list[-2], open('./data/dev_pred_%s_%s' % (args.train_mode, 'ALL'), 'wb'), -1)
            for i, j in mdev_pred_list.items():
                pickle.dump(mdev_pred_list[i][-2], open('./data/dev_pred_%s_%s' % (args.train_mode, i), 'wb'), -1)
            pickle.dump(key_mat_list[-1], open('./data/key_mat_%s_%s' % (args.train_mode, 'ALL'), 'wb'), -1)
            update_count = 0
        else:
            update_count += 1
        if update_count >= 3:
            exit()
        # shuffle train set ordering after every epoch
        random.shuffle(train_data)

    return

def evaluate_model_mul(args, tokenizer, model, mdev_pred_list, key_mat):
    print('Evaluating WSD model on {}...'.format(args.split))

    '''
    LOAD TRAINED MODEL
    '''
    mdataset = ['semeval2013-de',
                'semeval2013-es',
                'semeval2013-fr',
                'semeval2013-it',
                'semeval2015-es',
                'semeval2015-it'][:]
    correct, pred_all = 0, 0

    print('args.current_epoch', args.current_epoch)
    same_mode = args.same
    word_mode = args.word
    split = args.split
    for mdata in mdataset:
        args.split = mdata
        lang = mdata.split('-')[1]
        inventory_path = os.path.join(args.data_path, '{}/inventory.{}.withgold.wnids.txt'.format('inventories', lang))
        inventory = {i.strip().split('\t')[0]: i.strip().split('\t')[1:] for i in open(inventory_path).readlines()}

        eval_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/'.format(mdata))
        eval_data = load_data(eval_path, mdata)

        gloss_dict = load_bn_glosses(eval_data, tokenizer, inventory, lang, max_len=512)

        eval_data, dev_dict = preprocess_context(tokenizer, eval_data, gloss_dict, bsz=args.context_bsz,
                                                 max_len=args.context_max_length)

        dev_index = {}
        if args.current_epoch == 1:
            mdev_pred_list[mdata].append({})
            args.gloss_mode = 'non'

        args.same = False
        args.word = 'word'
        eval_preds, dev_index, key_mat = _eval(eval_data, model, gloss_dict, dev_index, dev_dict, key_mat)
        mdev_pred_list[mdata].append(deepcopy(dev_index))
        args.gloss_mode = 'sense-pred'

        #generate predictions file
        pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(mdata))
        with open(pred_filepath, 'w') as f:
            for inst, prediction in eval_preds:
                f.write('{} {}\n'.format(inst, prediction))

        #run predictions through scorer
        gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(mdata))
        scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
        p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
        correct += len(eval_preds) * f1 / 100
        pred_all += len(eval_preds)
        print('f1 of BERT probe on {} test set = {}'.format(mdata, f1))
    open('./data/dev_result.txt', 'a+').write('%s-%f\n' % (args.train_mode, correct/pred_all))
    print('mul_ALL', correct/pred_all)
    args.split = split
    args.same = same_mode
    args.word = word_mode
    return mdev_pred_list

def evaluate_model(args):
    print('Evaluating WSD model on {}...'.format(args.split))

    '''
    LOAD TRAINED MODEL
    '''
    if 'mul' in args.train_mode:
        mdataset = ['semeval2013-de',
                    'semeval2013-es',
                    'semeval2013-fr',
                    'semeval2013-it',
                    'semeval2015-es',
                    'semeval2015-it'][:]
    else:
        mdataset = ['ALL']
    correct, pred_all = 0, 0
    for mdata in mdataset:
        args.split = mdata
        if '-' in mdata:
            lang = mdata.split('-')[1]
            inventory_path = os.path.join(args.data_path, '{}/inventory.{}.withgold.wnids.txt'.format('inventories', lang))
            inventory = {i.strip().split('\t')[0]: i.strip().split('\t')[1:] for i in open(inventory_path).readlines()}

            print('inventory', len(inventory))
            eval_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/'.format(mdata))
            eval_data = load_data(eval_path, mdata)

            tokenizer = load_tokenizer(args.encoder_name)
            gloss_dict = load_bn_glosses(eval_data, tokenizer, inventory, lang, args.gloss_max_length)

            eval_data, dev_dict = preprocess_context(tokenizer, eval_data, gloss_dict, bsz=args.context_bsz,
                                                     max_len=args.context_max_length)

        else:
            '''
            LOAD TOKENIZER
            '''
            tokenizer = load_tokenizer(args.encoder_name)

            '''
            LOAD EVAL SET
            '''
            eval_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/'.format(args.split))
            eval_data = load_data(eval_path, args.split)

            #load gloss dictionary (all senses from wordnet for each lemma/pos pair that occur in data)
            wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
            wn_senses = load_wn_senses(wn_path)
            gloss_dict = load_and_preprocess_glosses(eval_data, tokenizer, wn_senses, max_len=args.gloss_max_length)

            eval_data, dev_dict = preprocess_context(tokenizer, eval_data, gloss_dict, bsz=args.context_bsz,
                                                     max_len=args.context_max_length)

        model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context)
        model_path = os.path.join(args.ckpt, 'best_model_%s.ckpt' % args.train_mode)
        model.load_state_dict(torch.load(model_path))
        model = model.cuda().half()
        '''
        EVALUATE MODEL
        '''
        # args.gloss_mode = 'sense'
        dev_index = pickle.load(open('./data/dev_pred_%s_%s' % (args.train_mode, mdata), 'rb'))

        key_mat = {}
        if args.sec_wsd:
            if os.path.exists('./data/key_mat_%s_%s' % (args.train_mode, 'ALL')):
                key_mat = pickle.load(open('./data/key_mat_%s_%s' % (args.train_mode, 'ALL'), 'rb'))
        eval_preds, dev_index, _ = _eval(eval_data, model, gloss_dict, dev_index, dev_dict, key_mat)

        #generate predictions file
        pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(args.split))
        with open(pred_filepath, 'w') as f:
            for inst, prediction in eval_preds:
                f.write('{} {}\n'.format(inst, prediction))

        #run predictions through scorer
        gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
        scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
        p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
        open('./data/dev_result.txt', 'a+').write('%f\n' % f1)
        print('f1 of BERT probe on {} test set = {}'.format(args.split, f1))
        correct += len(eval_preds) * f1 / 100
        pred_all += len(eval_preds)
    print('Average', correct/pred_all)
    return

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()

    #parse args
    args = parser.parse_args()
    print(args)

    #set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    #evaluate model saved at checkpoint or...
    if args.eval: evaluate_model(args)
    #train model
    else: train_model(args)

#EOF