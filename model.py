import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig
import random
import functools
from torchcrf import CRF
import gc
import math
import numpy as np
import os
import re
import collections
import transformer
import copy
import pickle
from utils import pad_sequence_right, pad_sequence_left

def multilabel_categorical_crossentropy(y_true, y_pred):
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)
    assert y_true.shape == y_pred.shape
    if len(y_true.shape) == 0:
        y_true = y_true.unsqueeze(0)
        y_pred = y_pred.squeeze(0)
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    #print(y_pred.shape)
    zeros = torch.zeros_like(y_pred[..., :1], device=y_pred.device)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


class MentionTypeEncoder(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super(MentionTypeEncoder, self).__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids, fuse_method='add'):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids, dtype=torch.long, device=batch_mention_emb.device, requires_grad=False
            )
        
        if fuse_method == 'add':
            batch_mention_type_emb = self.embedding(mention_type_ids)
            out = batch_mention_emb + batch_mention_type_emb
            out = self.dropout(self.layer_norm(out))
            return out
        elif fuse_method == 'concat':
            batch_mention_type_emb = self.embedding(mention_type_ids)
            out = torch.cat([batch_mention_emb, batch_mention_type_emb], dim=-1)
            return out

class EventTable(nn.Module):
    def __init__(self, event_type, field_types, hidden_size, output_type='logp_logits'):
        super(EventTable, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.hidden_size = hidden_size
        self.output_type = output_type

        if output_type.endswith('logits'):
            self.event_cls = nn.Linear(hidden_size, 2)  # 0: NA, 1: trigger this event
            self.field_cls_list = nn.ModuleList(
                # 0: NA, 1: trigger this field
                [nn.Linear(hidden_size, 2) for _ in range(self.num_fields)]
            )
        elif output_type == 'score':
            self.event_cls = nn.Linear(hidden_size, 1)  # 0: NA, 1: trigger this event
            self.field_cls_list = nn.ModuleList(
                # 0: NA, 1: trigger this field
                [nn.Linear(hidden_size, 1) for _ in range(self.num_fields)]
            )

        # used to aggregate sentence and span embedding
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for fields that do not contain any valid span
        # self.none_span_emb = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for aggregating history filled span info
        self.field_queries = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.hidden_size)) for _ in range(self.num_fields)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        # self.none_span_emb.data.uniform_(-stdv, stdv)
        for fq in self.field_queries:
            fq.data.uniform_(-stdv, stdv)
    
    def forward(self, sent_context_emb=None, batch_span_emb=None, field_idx=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = transformer.attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            if self.output_type == 'logits' or self.output_type == 'score':
                return doc_pred_logits
            elif self.output_type == 'logp_logits':
                doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)
                return doc_pred_logp

        if batch_span_emb is not None:
            assert field_idx is not None
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            span_pred_logits = self.field_cls_list[field_idx](batch_span_emb)
            if self.output_type == 'logits' or self.output_type == 'score':
                return span_pred_logits
            elif self.output_type == 'logp_logits':
                span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)
                return span_pred_logp
        
class SentencePosEncoder(nn.Module):
    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))
        return out

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def get_pe(self, p):
        return self.pe[:, p].squeeze()

    def forward(self, x, pos_ids_list, fuse_method='add'):
        if fuse_method == 'add':
            x = x + Variable(self.pe[:, pos_ids_list], requires_grad=False).squeeze(0)
            return self.dropout(x)
        elif fuse_method == 'concat':
            pos_emb = Variable(self.pe[:, pos_ids_list], requires_grad=False).squeeze(0)
            x = torch.cat([x, pos_emb], dim=-1)
            return x
        # x = x + Variable(self.pe[:, :x.size(1)], 
        #                  requires_grad=False)
        # return self.dropout(x)

class AttentiveReducer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        return 'hidden_size={}, att_norm={}'.format(self.hidden_size, self.att_norm)

class AttentionVal(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionVal, self).__init__()
        self.q_map = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        self.k_map = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )

    # expect vec_seq.shape as (num_vec, hidden_size)
    def forward(self, vec_seq):
        q = self.q_map(vec_seq)
        k = self.k_map(vec_seq)
        att = torch.matmul(q, k.transpose(0, 1)) # (num_vec, num_vec)
        att = torch.sigmoid(att)
        return att

class DocEE(nn.Module):
    def __init__(self, config, basic_encoder, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.init_eval_obj()
        self.basic_encoder = basic_encoder

        hidden_size = config['hidden_size']
        dropout = config['dropout']

        if config['xlnet_doc_bidirection']:
            # hidden_size *= 2
            self.re_basic_encoder = copy.deepcopy(basic_encoder)
            self.bi_linear_map = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

        if config['use_crf']:
            self.crf = CRF(num_tags=len(self.config['NER_LABEL_LIST']), batch_first=True)

        self.seq_labeler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, len(self.config['NER_LABEL_LIST']))
        )

        if config['use_rnn_enc'] is not None:
            rnn_bidirection = self.config['rnn_bidirection']
            rnn_hidden_size = hidden_size
            if rnn_bidirection:
                rnn_hidden_size = rnn_hidden_size // 2
            if config['use_rnn_enc'] == 'LSTM':
                self.rnn_encoder = nn.LSTM(
                    input_size=hidden_size, hidden_size=rnn_hidden_size, num_layers=2, dropout=dropout, batch_first=True,
                    bidirectional=rnn_bidirection
                )
            elif config['use_rnn_enc'] == 'GRU':
                self.rnn_encoder = nn.GRU(
                    input_size=hidden_size, hidden_size=rnn_hidden_size, num_layers=2, dropout=dropout, batch_first=True,
                    bidirectional=rnn_bidirection,
                )

        if config['cut_word_task']:
            self.cw_labeler = nn.Linear(hidden_size, 2) # only 0 or 1
        
        if config['pos_tag_task']:
            self.pos_tag_labeler = nn.Linear(hidden_size, len(config['POS_TAG_LIST']))
        
        if config['parser_task']:
            self.parser_query = nn.Linear(hidden_size, hidden_size // 2)
            self.parser_key = nn.Linear(hidden_size, hidden_size // 2)

        self.event_tables = []
        for event_type in self.config['EVENT_TYPES']:
            if config['multilabel_loss'] == 'binary':
                event_table = EventTable(event_type, self.config['EVENT_FIELDS'][event_type][0], hidden_size)
            elif config['multilabel_loss'] == 'multilabel_crossentropy':
                event_table = EventTable(event_type, self.config['EVENT_FIELDS'][event_type][0], hidden_size, output_type='score')
            self.event_tables.append(event_table)
        self.event_tables = nn.ModuleList(self.event_tables)

        # self.event_tables = nn.ModuleList([
        #     EventTable(event_type, self.config['EVENT_FIELDS'][event_type][0], hidden_size)
        #     for event_type in self.config['EVENT_TYPES']
        # ])

        if config['use_pos_emb']:
            if config['trainable_pos_emb']:
                self.sent_pos_encoder = SentencePosEncoder(
                    hidden_size, max_sent_num=config['max_sent_num'], dropout=dropout
                )
            else:
                self.sent_pos_encoder = PositionalEncoding(hidden_size, dropout)

        if config['use_token_role']:
            self.ment_type_encoder = MentionTypeEncoder(
                hidden_size, len(config['NER_LABEL_LIST']), dropout=dropout
            )

        if config['use_doc_enc']:
            self.doc_context_encoder = transformer.make_transformer_encoder(
                config['num_tf_layer'], hidden_size,
                ff_size=config['ff_size'], dropout=dropout
            )

        if config['use_edag_graph']:
            self.span2span_att = AttentionVal(hidden_size)

        self.field_context_encoder = transformer.make_transformer_encoder(
            config['num_tf_layer'], hidden_size, ff_size=config['ff_size'], dropout=dropout
        )

        if config['span_drange_fuse'] == 'concat':
            fuse_input_dim = hidden_size * 2
            if self.config['use_token_role']:
                fuse_input_dim += hidden_size
            self.drange_fuser = nn.Sequential(
                nn.Linear(fuse_input_dim, fuse_input_dim // 2),
                nn.ReLU(),
                nn.Linear(fuse_input_dim // 2, hidden_size),
                nn.Dropout(dropout)
            )

        if config['pooling'] == 'AWA':
            self.sent_ids_reducer = AttentiveReducer(hidden_size, dropout=dropout)
            self.ids_reducer = AttentiveReducer(hidden_size, dropout=dropout)
            self.drange_reducer = AttentiveReducer(hidden_size, dropout=dropout)

    def init_eval_obj(self):
        self.eval_obj = {
            'event_type_pred': [], 'event_type_gt': [], 'ner_pred': [], 'ner_gt': [],
            'span2span_pred': [], 'span2span_gt': [], 'ins': [], 'decode_res': [],
        }

    def pooling(self, emb, attention_mask=None, dim=0, att_reducer=None):
        # emb.shape should be (batch_size, sent_length, hidden_size)
        pooling_type = self.config['pooling']
        if pooling_type == 'max':
            if attention_mask is not None:
                emb = emb.masked_fill(attention_mask == 0, -float('inf'))
            pooling_emb = torch.max(emb, dim=dim)[0]
        elif pooling_type == 'mean':
            if attention_mask is not None:
                emb = emb.masked_fill(attention_mask == 0, 0)
                pooling_emb = torch.sum(emb, dim=dim) / torch.sum(attention_mask, dim=dim)
            else:
                pooling_emb = torch.mean(emb, dim=dim)
        elif pooling_type == 'AWA':
            if dim == 0:
                pooling_emb = att_reducer(emb.unsqueeze(0) , masks=attention_mask).squeeze()
            elif dim == 1:
                pooling_emb = att_reducer(emb, masks=attention_mask.squeeze())
            else:
                raise Exception('AttentiveReducer does not support such dim')
        else:
            raise Exception('Not support pooling method: ' + pooling_type)
        return pooling_emb

    def forward(self, batch_train, train_flag=True, dev_flag=False, use_gold=True):
        ee_method = self.config['ee_method']
        #ee_method = 'GreedyDec'

        ner_loss, doc_ids_emb, doc_ner_pred, doc_sent_emb, doc_ner_score = self.do_ner(batch_train, train_flag, use_gold)
        if self.config['only_ner']:
            return ner_loss, []

        if ee_method == 'GreedyDec':
            decode_loss, decode_res = self.greedy_dec(doc_sent_emb, doc_ner_pred, batch_train, train_flag)
            total_loss = ner_loss + decode_loss
            return total_loss, decode_res
        elif ee_method == 'EDAG':
            doc_span_drange_list, doc_span_emb, doc_sent_emb = self.pooling_for_span(
                doc_ids_emb, doc_ner_pred, doc_sent_emb, batch_train, train_flag=train_flag
            )
            doc_span2span_info = None
            doc_edag_info = None
            if train_flag or dev_flag:
                doc_edag_info, doc_span2span_info = self.generate_train_edag(doc_span_drange_list, batch_train)
                
            decode_loss, decode_res, doc_score_info = self.edag_dec(
                doc_span_drange_list, doc_span_emb, doc_sent_emb, doc_edag_info,
                batch_train, train_flag=train_flag, use_gold=use_gold, doc_span2span_info=doc_span2span_info
            )
            total_loss = ner_loss + 0.3 * decode_loss
            return total_loss, decode_res
        else:
            raise Exception('Not support event method: ' + ee_method)

    def edag_dec(self, doc_span_drange_list, doc_span_emb, doc_sent_emb, doc_edag_info, batch_train,\
        train_flag=True, use_gold=True, doc_span2span_info=None):
        doc_decode_res = []
        event_cls_loss = 0
        doc_edag_loss = 0
        span2span_loss = 0
        valid_edag_count = 0
        doc_score_info = []
        for i, (span_drange_list, span_emb, sent_emb, ins) in enumerate(zip(
            doc_span_drange_list, doc_span_emb, doc_sent_emb, batch_train
        )):
            score_info = {}
            self.eval_obj['ins'].append(ins)

            event_type_score = []
            for event_table in self.event_tables:
                event_type_score.append(event_table(sent_context_emb=sent_emb)) # already log softmax
            event_type_score = torch.cat(event_type_score, dim=0)
            score_info['event_type_score'] = np.exp(event_type_score.detach().cpu().numpy()).reshape(-1)

            #event cls
            if train_flag:
                if self.config['multilabel_loss'] == 'binary':
                    score_info['event_type_target'] = ins['event_cls']
                    event_type_label = torch.tensor(ins['event_cls'], device=event_type_score.device)
                    event_cls_loss += F.nll_loss(event_type_score, event_type_label)
                    event_type_pred = ins['event_cls'] # edag training always uses gt event cls labels
                elif self.config['multilabel_loss'] == 'multilabel_crossentropy':
                    score_info['event_type_target'] = ins['event_cls']
                    event_type_label = torch.tensor(ins['event_cls'], device=event_type_score.device, dtype=torch.float)
                    event_cls_loss += multilabel_categorical_crossentropy(event_type_label, event_type_score)
                    event_type_pred = ins['event_cls'] # edag training always uses gt event cls labels
            else:
                if self.config['multilabel_loss'] == 'binary':
                    event_type_pred = torch.argmax(event_type_score, dim=-1).tolist()
                elif self.config['multilabel_loss'] == 'multilabel_crossentropy':
                    event_type_pred = (event_type_score > 0).squeeze().tolist()
            
            if span_drange_list is None or len(span_drange_list) < 1:
                doc_decode_res.append([None for _ in range(len(self.event_tables))])
                continue
            
            span2span_score = None
            if self.config['use_edag_graph']:
                span2span_score = self.span2span_att(span_emb)
                if doc_span2span_info is not None:
                    self.eval_obj['span2span_pred'].append(span2span_score.tolist())
                    self.eval_obj['span2span_gt'].append(doc_span2span_info[i])
                if train_flag:
                    span2span_target = torch.tensor(doc_span2span_info[i], dtype=torch.float, device=span2span_score.device, requires_grad=False)
                    span2span_loss += torch.nn.functional.mse_loss(span2span_score, span2span_target)
                if use_gold:
                    span2span_score = span2span_target
                
            edag_loss = 0
            decode_res = []
            score_info['edag_score_info'] = []
            for event_idx, event_cls_label in enumerate(event_type_pred):
                if train_flag:
                    if event_cls_label == 0:
                        edag_field_loss, edag_score_info = self.field_loss_by_edag(
                            span_emb, sent_emb, event_idx, edag_info=None, span2span_score=None
                        )
                        edag_loss += edag_field_loss
                    else:
                        assert doc_edag_info[i][event_idx] is not None
                        edag_field_loss, edag_score_info = self.field_loss_by_edag(
                            span_emb, sent_emb, event_idx, edag_info=doc_edag_info[i][event_idx],
                            span2span_score=span2span_score
                        )
                        edag_loss += edag_field_loss
                    score_info['edag_score_info'].append(edag_score_info)
                else:
                    if event_cls_label == 0:
                        decode_res.append(None)
                    else:
                        decode_paths = self.infer_field_by_edag(span_emb, sent_emb, event_idx, span2span_score=span2span_score)
                        assert len(decode_paths) > 0
                        
                        res = []
                        for decode_path in decode_paths:
                            field_res = []
                            for span_idx in decode_path:
                                if span_idx is not None:
                                    field_res.append(span_drange_list[span_idx][1])
                                else:
                                    field_res.append(None)
                            res.append(field_res)
                        decode_res.append(res)
            self.eval_obj['decode_res'].append(decode_res)
            doc_decode_res.append(decode_res)
            doc_score_info.append(score_info)

            edag_loss /= len(event_type_pred)
            doc_edag_loss += edag_loss
            valid_edag_count += 1

        doc_edag_loss = (doc_edag_loss / valid_edag_count) if valid_edag_count > 0 else doc_edag_loss
        event_cls_loss /= len(batch_train)
        span2span_loss /= len(batch_train)
        return event_cls_loss + doc_edag_loss + 0.3 * span2span_loss, doc_decode_res, doc_score_info
    
    def infer_field_by_edag(self, span_emb, sent_emb, event_idx, span2span_score=None):
        device = span_emb.device
        event_type = self.config['EVENT_TYPES'][event_idx]
        ordered_fields = self.config['EVENT_FIELDS'][event_type][0]
        num_fields = len(ordered_fields)
        num_spans = span_emb.shape[0]
        event_table = self.event_tables[event_idx]

        cur_mem_list = []
        use_edag_graph = self.config['use_edag_graph']
        if use_edag_graph:
            cur_span_graph_score = torch.ones(len(span_emb), dtype=torch.float, device=device)
            cur_mem_list.append(span_emb)
        cur_mem_list.append(sent_emb)
        prev_path2mem = {
            (): cur_mem_list
        }
        # prev_path2mem = {
        #     (): sent_emb
        # }
        last_field_paths = [()]
        for field_idx in range(num_fields):
            cur_paths = []
            for prev_path in last_field_paths:
                assert prev_path in prev_path2mem
                prev_mem_list = prev_path2mem[prev_path]
                prev_mem = torch.cat(prev_mem_list, dim=0)
                #prepare for input
                field_query = event_table.field_queries[field_idx]
                span_cand_emb = span_emb + field_query
                cand_emb = torch.cat([span_cand_emb, prev_mem], dim=0).unsqueeze(0)
                #pred
                cand_output = self.field_context_encoder(cand_emb, None).squeeze(0)
                span_output = cand_output[:num_spans]
                span_score = event_table(batch_span_emb=span_output, field_idx=field_idx) # already log softmax

                if span_score.shape[0] > 100:
                    span_score = span_score[:20]

                if self.config['multilabel_loss'] == 'binary':
                    span_logp = span_score
                    span_pred = torch.argmax(span_logp, dim=-1).tolist()
                elif self.config['multilabel_loss'] == 'multilabel_crossentropy':
                    span_pred = (span_score > 0).squeeze().tolist()
                    if not isinstance(span_pred, list):
                        span_pred = [span_pred]
                
                #update path
                cur_span_idx_list = []
                for cur_span_idx, pred_label in enumerate(span_pred):
                    if pred_label == 1:
                        cur_span_idx_list.append(cur_span_idx)
                if len(cur_span_idx_list) == 0:
                    cur_span_idx_list.append(None)
                
                for span_idx in cur_span_idx_list:
                    cur_path = prev_path + (span_idx, )
                    if span_idx is None:
                        span_mem = field_query
                    else:
                        span_mem = span_output[span_idx].unsqueeze(0)

                    cur_mem_list = list(prev_mem_list) #copy
                    if use_edag_graph: # the first memory should be span2span memory
                        if span_idx is not None:
                            cur_span_graph_score += span2span_score[span_idx]
                        else:
                            cur_span_graph_score += 1
                        cur_mem_list[0] = span_emb * (cur_span_graph_score.view(-1, 1) / (field_idx + 2))
                    cur_mem_list.append(span_mem)
                    prev_path2mem[cur_path] = cur_mem_list
                    # cur_mem = torch.cat([prev_mem, span_mem], dim=0)
                    # prev_path2mem[cur_path] = cur_mem
                    cur_paths.append(cur_path)
                    
            if len(cur_paths) > 50:
                cur_paths = cur_paths[:5]
            last_field_paths = cur_paths
        return last_field_paths

    def field_loss_by_edag(self, span_emb, sent_emb, event_idx, edag_info=None, span2span_score=None):
        device = span_emb.device
        event_type = self.config['EVENT_TYPES'][event_idx]
        ordered_fields = self.config['EVENT_FIELDS'][event_type][0]
        num_fields = len(ordered_fields)
        num_spans = span_emb.shape[0]
        event_table = self.event_tables[event_idx]
        class_weight = torch.tensor(
            [self.config['neg_field_loss_scaling'], 1.0], device=device, dtype=torch.float, requires_grad=False
        )
        field_loss = 0
        score_target_info = {}
        if edag_info is None: # negative case
            # prev_mem = sent_emb
            # span_label = torch.tensor([0 for _ in range(num_spans)], dtype=torch.long, device=device, requires_grad=False)
            # for field_idx in range(num_fields):
            #     #prepare for input
            #     field_query = event_table.field_queries[field_idx]
            #     span_cand_emb = span_emb + field_query
            #     cand_emb = torch.cat([span_cand_emb, prev_mem], dim=0).unsqueeze(0)

            #     #train
            #     cand_output = self.field_context_encoder(cand_emb, None).squeeze(0)
            #     span_output = cand_output[:num_spans]
            #     span_logp = event_table(batch_span_emb=span_output, field_idx=field_idx) # already log softmax
            #     field_loss += F.nll_loss(span_logp, span_label, weight=class_weight)
                
            #     #update_memory
            #     prev_mem = torch.cat([prev_mem, field_query], dim=0)
            pass
        else: # positive case
            cur_mem_list = []
            use_edag_graph = self.config['use_edag_graph']
            if use_edag_graph:
                cur_span_graph_score = torch.ones(len(span_emb), dtype=torch.float, device=device)
                cur_mem_list.append(span_emb)
            cur_mem_list.append(sent_emb)
            prev_path2mem = {
                (): cur_mem_list
            }
            # prev_path2mem = {
            #     (): sent_emb
            # }
            for field_idx in range(num_fields):
                prev_path2cur_span_idx_set = edag_info[field_idx]
                for prev_path, cur_span_idx_set in edag_info[field_idx].items():
                    assert prev_path in prev_path2mem
                    prev_mem_list = prev_path2mem[prev_path]
                    prev_mem = torch.cat(prev_mem_list, dim=0)

                    #prepare for input
                    field_query = event_table.field_queries[field_idx]
                    span_cand_emb = span_emb + field_query
                    cand_emb = torch.cat([span_cand_emb, prev_mem], dim=0).unsqueeze(0)
                    #train
                    cand_output = self.field_context_encoder(cand_emb, None).squeeze(0)
                    span_output = cand_output[:num_spans]
                    span_score = event_table(batch_span_emb=span_output, field_idx=field_idx)

                    span_label = [1 if span_idx in cur_span_idx_set else 0 for span_idx in range(num_spans)]
                    # score_target_info[prev_path] = {
                    #     'logp_score': np.exp(span_logp.detach().cpu().numpy()),
                    #     'target': span_label,
                    #     'field_idx': field_idx,
                    #     'event_idx': event_idx
                    # }
                    if self.config['multilabel_loss'] == 'binary':
                        span_logp = span_score
                        span_label = torch.tensor(span_label, dtype=torch.long, device=device, requires_grad=False)
                        field_loss += F.nll_loss(span_logp, span_label, weight=class_weight)
                    elif self.config['multilabel_loss'] == 'multilabel_crossentropy':
                        span_label = torch.tensor(span_label, dtype=torch.float, device=device, requires_grad=False)
                        field_loss += multilabel_categorical_crossentropy(span_label, span_score)

                    #update_memory
                    for span_idx in cur_span_idx_set:
                        cur_path = prev_path + (span_idx, )
                        if span_idx is None:
                            span_mem = field_query
                        else:
                            span_mem = span_output[span_idx].unsqueeze(0)
                        cur_mem_list = list(prev_mem_list) #shallow copy
                        if use_edag_graph: # the first memory should be span2span memory
                            if span_idx is not None:
                                cur_span_graph_score += span2span_score[span_idx]
                            else:
                                cur_span_graph_score += 1
                            cur_mem_list[0] = span_emb * (cur_span_graph_score.view(-1, 1) / (field_idx + 2))
                        cur_mem_list.append(span_mem)
                        prev_path2mem[cur_path] = cur_mem_list
                        #cur_mem = torch.cat([prev_mem, span_mem], dim=0)
                        #prev_path2mem[cur_path] = cur_mem
        return field_loss / num_fields, score_target_info

    def pooling_for_span(self, doc_ids_emb, doc_ner_pred, doc_sent_emb, batch_train, doc_enc=False, train_flag=False):
        NER_LABEL_LIST = self.config['NER_LABEL_LIST']
        doc_span_drange_list = []
        doc_span_emb = []
        res_doc_sent_emb = []
        for ids_emb, ner_pred, sent_emb, ins in zip(doc_ids_emb, doc_ner_pred, doc_sent_emb, batch_train):

            ids2drange = collate_label(ner_pred, ins['attention_mask'], ins['ids_list'])
            if len(ids2drange) == 0:
                if train_flag: # have to use gold label
                    ids2drange = collate_label(ins['labels_list'], ins['attention_mask'], ins['ids_list'])
                else:
                    doc_span_drange_list.append(None)
                    doc_span_emb.append(None)
                    if self.config['use_doc_enc']:
                        sent_emb = self.doc_context_encoder(sent_emb.unsqueeze(0), None).squeeze(0)
                    res_doc_sent_emb.append(sent_emb)
                    continue

            span2drange = {}
            # pooling by token
            for ids, dranges in ids2drange.items():
                for drange, ner_label_idx in dranges:
                    sent_idx, char_s, char_e = drange
                    span = ins['merge_sentences'][sent_idx][char_s: char_e]

                    reducer = None
                    if self.config['pooling'] == 'AWA':
                        reducer = self.ids_reducer
                    span_emb = self.pooling(ids_emb[sent_idx, char_s: char_e], dim=0, att_reducer=reducer)

                    ner_label = NER_LABEL_LIST[ner_label_idx]
                    assert ner_label.startswith('B-')
                    if span in span2drange:
                        span2drange[span].append((drange, ner_label_idx, span_emb))
                    else:
                        span2drange[span] = [(drange, ner_label_idx, span_emb)]
            span_list = list(span2drange.keys())
            span_beg_list = [0]
            drange_list = []
            drange_sent_pos_list = []
            drange_ner_label_list = []
            batch_drange_emb = []
            for dranges in span2drange.values():
                for drange, ner_label_idx, span_emb in dranges:
                    drange_list.append(drange)
                    drange_sent_pos_list.append(drange[0])
                    drange_ner_label_list.append(ner_label_idx)
                    batch_drange_emb.append(span_emb)
                span_beg_list.append(len(drange_list))
                
            batch_drange_emb = torch.stack(batch_drange_emb, dim=0)
            fuse_method = self.config['span_drange_fuse']
            if self.config['use_pos_emb']:
                batch_drange_emb = self.sent_pos_encoder(batch_drange_emb, drange_sent_pos_list, fuse_method=fuse_method)
            if self.config['use_token_role']:
                batch_drange_emb = self.ment_type_encoder(batch_drange_emb, drange_ner_label_list, fuse_method=fuse_method)
            
            if fuse_method == 'concat':
                batch_drange_emb = self.drange_fuser(batch_drange_emb)
            
            batch_span_emb = []
            span_drange_list = []
            if self.config['use_doc_enc']:
                doc_ctx = torch.cat([batch_drange_emb, sent_emb], dim=0).unsqueeze(0)
                doc_ctx = self.doc_context_encoder(doc_ctx, None).squeeze(0)
                batch_drange_emb = doc_ctx[:batch_drange_emb.shape[0]]
                sent_emb = doc_ctx[batch_drange_emb.shape[0]:]

            for span_i, span in enumerate(span_list):
                span_beg = span_beg_list[span_i]
                span_end = span_beg_list[span_i + 1]

                reducer = None
                if self.config['pooling'] == 'AWA':
                    reducer = self.drange_reducer
                span_emb = self.pooling(batch_drange_emb[span_beg: span_end], att_reducer=reducer)
                batch_span_emb.append(span_emb)
                span_drange_list.append((span, drange_list[span_beg]))

            batch_span_emb = torch.stack(batch_span_emb, dim=0)
            doc_span_emb.append(batch_span_emb)
            res_doc_sent_emb.append(sent_emb)
            doc_span_drange_list.append(span_drange_list)
        
        return doc_span_drange_list, doc_span_emb, res_doc_sent_emb

    def greedy_dec(self, doc_sent_emb, doc_ner_pred, batch_train, train_flag=True):
        doc_decode_res = []
        event_cls_loss = 0
        for sent_emb, ner_pred, ins in zip(doc_sent_emb, doc_ner_pred, batch_train):
            event_type_score = []
            for event_table in self.event_tables:
                 event_type_score.append(event_table(sent_context_emb=sent_emb))
            event_type_score = torch.cat(event_type_score, dim=0)
            
            if train_flag:
                event_type_label = torch.tensor(ins['event_cls'], device=event_type_score.device)
                event_cls_loss += F.nll_loss(event_type_score, event_type_label)
            else:
                ids2drange = collate_label(ner_pred, ins['attention_mask'], ins['ids_list'])
                event_type_pred = torch.argmax(event_type_score, dim=-1).tolist()
                self.eval_obj['event_type_pred'].append(event_type_pred)
                if ins.get('event_cls') is not None:
                    self.eval_obj['event_type_gt'].append(ins['event_cls'])
                # event_type_pred = ins['event_cls']
                
                EVENT_TYPES = self.config['EVENT_TYPES']
                EVENT_FIELDS = self.config['EVENT_FIELDS']
                NER_LABEL_LIST = self.config['NER_LABEL_LIST']
                label2drange = {}
                for ids, dranges in ids2drange.items():
                    for drange, label_idx in dranges:
                        label = NER_LABEL_LIST[label_idx]
                        #assert label.startswith('B-')
                        label = label[2:]
                        if label not in label2drange:
                            label2drange[label] = []
                        label2drange[label].append(drange)
                decode_res = []
                for event_idx, pred in enumerate(event_type_pred):
                    if pred == 0:
                        decode_res.append(None)
                        continue
                    event_type = EVENT_TYPES[event_idx]
                    fields = EVENT_FIELDS[event_type][0]
                    field_res = []
                    for field in fields:
                        if field not in label2drange:
                            field_res.append(None)
                            continue
                        field_res.append(label2drange[field][0]) # greedy
                    decode_res.append([field_res])
                doc_decode_res.append(decode_res)
        event_cls_loss /= len(doc_sent_emb) #mean for batch
        return event_cls_loss, doc_decode_res
        
    
    def do_ner(self, batch_train, train_flag=True, use_gold=False):
        device = self.basic_encoder.device
        input_ids = []
        ner_label = []
        attention_mask = []
        ids_length = []
        sent_pos_ids = []
        doc_beg_list = [0]

        cw_label = []
        pos_label = []
        parser_label = []

        input_ids_by_sent_idx = []
        attention_mask_by_sent_idx = []
        batch_idx_by_sent_idx = []
        cur_max_sent_num = 0
        for batch_idx, ins in enumerate(batch_train):
            if train_flag:
                ner_label.extend(ins['labels_list'])
                if self.config['cut_word_task']:
                    cw_label.extend(ins['cw_labels_list'])
                if self.config['pos_tag_task']:
                    pos_label.extend(ins['pos_tag_labels_list'])
                if self.config['parser_task']:
                    parser_label.extend(ins['parser_labels_list'])


            input_ids.extend(ins['ids_list'])
            attention_mask.extend(ins['attention_mask'])
            ids_length.extend(ins['ids_length'])
            sent_pos_ids.extend(list(range(len(ins['ids_list']))))
            doc_beg_list.append(len(input_ids))

            cur_max_sent_num = max(cur_max_sent_num, len(ins['ids_list']))
            for sent_idx, ids in enumerate(ins['ids_list']):
                if sent_idx >= len(input_ids_by_sent_idx):
                    input_ids_by_sent_idx.append([])
                    attention_mask_by_sent_idx.append([])
                    batch_idx_by_sent_idx.append([])
                input_ids_by_sent_idx[sent_idx].append(ids)
                attention_mask_by_sent_idx[sent_idx].append(ins['attention_mask'][sent_idx])
                batch_idx_by_sent_idx[sent_idx].append(batch_idx)

            self.eval_obj['ner_gt'].append(ins['labels_list'])
        attention_mask = torch.tensor(attention_mask, device=device, dtype=torch.float)
        
        if self.config['use_bert']:
            input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
            if self.config['sent_batch_size'] is not None:
                sent_batch_size = self.config['sent_batch_size']
                sent_batch_num = math.ceil(input_ids.shape[0] / sent_batch_size)
                batch_emb = []
                for sent_batch_i in range(sent_batch_num):
                    sent_batch_beg = sent_batch_i * sent_batch_size
                    sent_batch_end = (sent_batch_i + 1) * sent_batch_size
                    batch_input_ids = input_ids[sent_batch_beg: sent_batch_end]
                    if batch_input_ids.shape[0] < 1:
                        continue
                    batch_attention_mask = attention_mask[sent_batch_beg: sent_batch_end]
                    batch_emb.append(self.basic_encoder(batch_input_ids, attention_mask=batch_attention_mask)[0])
                batch_emb = torch.cat(batch_emb, dim=0)
            else:
                batch_emb = self.basic_encoder(input_ids, attention_mask=attention_mask)[0]
        elif self.config['use_xlnet']:
            mems = None
            re_mems = None
            batch_emb = []
            re_batch_emb = list(range(cur_max_sent_num))
            batch_input_ids_by_sent_idx = list(range(cur_max_sent_num))
            batch_attention_mask_by_sent_idx = list(range(cur_max_sent_num))
            for sent_idx in range(cur_max_sent_num):
                if not self.config['xlnet_doc_reverse']:
                    #forward
                    if isinstance(batch_input_ids_by_sent_idx[sent_idx], torch.Tensor):
                        batch_input_ids = batch_input_ids_by_sent_idx[sent_idx]
                        batch_attention_mask = batch_attention_mask_by_sent_idx[sent_idx]
                    else:
                        batch_input_ids = torch.tensor(input_ids_by_sent_idx[sent_idx], device=device, dtype=torch.long)
                        batch_attention_mask = torch.tensor(attention_mask_by_sent_idx[sent_idx], device=device, dtype=torch.float)
                        batch_input_ids_by_sent_idx[sent_idx] = batch_input_ids
                        batch_attention_mask_by_sent_idx[sent_idx] = batch_attention_mask

                    if mems is not None and mems[0].shape[1] > batch_input_ids.shape[0]:
                        abs_idx = []
                        for batch_idx in batch_idx_by_sent_idx[sent_idx]:
                            abs_idx.append(batch_idx_by_sent_idx[sent_idx - 1].index(batch_idx))

                        n_mems = []
                        for mem in mems:
                            n_mems.append(mem[:, tuple(abs_idx)])
                        mems = n_mems
                    emb, mems = self.basic_encoder(batch_input_ids, attention_mask=batch_attention_mask, mems=mems)
                    batch_emb.append(emb)

                #backward
                if self.config['xlnet_doc_bidirection'] or self.config['xlnet_doc_reverse']:
                    re_sent_idx = cur_max_sent_num - 1 - sent_idx
                    if isinstance(batch_input_ids_by_sent_idx[re_sent_idx], torch.Tensor):
                        batch_input_ids = batch_input_ids_by_sent_idx[re_sent_idx]
                        batch_attention_mask = batch_attention_mask_by_sent_idx[re_sent_idx]
                    else:
                        batch_input_ids = torch.tensor(input_ids_by_sent_idx[re_sent_idx], device=device, dtype=torch.long)
                        batch_attention_mask = torch.tensor(attention_mask_by_sent_idx[re_sent_idx], device=device, dtype=torch.float)
                        batch_input_ids_by_sent_idx[re_sent_idx] = batch_input_ids
                        batch_attention_mask_by_sent_idx[re_sent_idx] = batch_attention_mask

                    if re_mems is not None and re_mems[0].shape[1] < batch_input_ids.shape[0]:
                        o_abs_idx = []
                        for batch_idx in batch_idx_by_sent_idx[re_sent_idx + 1]:
                            o_abs_idx.append(batch_idx_by_sent_idx[re_sent_idx].index(batch_idx))
                        n_abs_idx = [i for i in range(len(batch_idx_by_sent_idx[re_sent_idx])) if i not in o_abs_idx]
                        o_emb, o_mems = self.basic_encoder(batch_input_ids[o_abs_idx], attention_mask=batch_attention_mask[o_abs_idx], mems=re_mems)
                        n_emb, n_mems = self.basic_encoder(batch_input_ids[n_abs_idx], attention_mask=batch_attention_mask[n_abs_idx], mems=None)

                        re_mems = [list(range(len(batch_idx_by_sent_idx[re_sent_idx]))) for _ in re_mems]
                        re_emb = list(range(len(batch_idx_by_sent_idx[re_sent_idx])))
                        for emb_idx, idx in enumerate(o_abs_idx):
                            re_emb[idx] = o_emb[emb_idx]
                            for l, layer_mem in enumerate(o_mems):
                                re_mems[l][idx] = layer_mem[:, emb_idx]
                        for emb_idx, idx in enumerate(n_abs_idx):
                            re_emb[idx] = n_emb[emb_idx]
                            for l, layer_mem in enumerate(n_mems):
                                re_mems[l][idx] = layer_mem[:, emb_idx]
                        
                        re_emb = torch.stack(re_emb, dim=0)
                        for l, layer_mem in enumerate(re_mems):
                            layer_mem = pad_sequence_right(layer_mem, batch_first=False, padding_value=0)
                            re_mems[l] = layer_mem
                        re_batch_emb[re_sent_idx] = re_emb
                    else:
                        re_emb, re_mems = self.basic_encoder(batch_input_ids, attention_mask=batch_attention_mask, mems=re_mems)
                        re_batch_emb[re_sent_idx] = re_emb

            n_batch_emb = []
            for batch_idx in range(len(batch_train)):
                ins = batch_train[batch_idx]
                for sent_idx in range(len(ins['ids_list'])):
                    abs_idx = batch_idx_by_sent_idx[sent_idx].index(batch_idx)
                    if self.config['xlnet_doc_bidirection']:
                        n_batch_emb.append(torch.cat([batch_emb[sent_idx][abs_idx], re_batch_emb[sent_idx][abs_idx]], dim=-1))
                        # n_batch_emb.append(re_batch_emb[sent_idx][abs_idx])
                    elif self.config['xlnet_doc_reverse']:
                        n_batch_emb.append(re_batch_emb[sent_idx][abs_idx])
                    else:
                        n_batch_emb.append(batch_emb[sent_idx][abs_idx])
            batch_emb = torch.stack(n_batch_emb, dim=0)

            if self.config['xlnet_doc_bidirection']:
                batch_emb = self.bi_linear_map(batch_emb)

        ner_score = self.seq_labeler(batch_emb)

        if self.config['use_rnn_enc'] is not None:
            rnn_batch_emb = []
            rnn_batch_input = []

            for i in range(len(doc_beg_list) - 1):
                doc_beg, doc_end = doc_beg_list[i], doc_beg_list[i + 1]
                ids_emb = batch_emb[doc_beg: doc_end]
                rnn_mem = None
                rnn_input = []
                for sent_idx in range(ids_emb.shape[0]):
                    length = ids_length[doc_beg: doc_end][sent_idx]
                    rnn_input.append(ids_emb[sent_idx, :length])
                rnn_batch_input.append(torch.cat(rnn_input))
            pack_seq = nn.utils.rnn.pack_sequence(rnn_batch_input, enforce_sorted=False)
            pack_output = self.rnn_encoder(pack_seq)[0]
            rnn_batch_emb = nn.utils.rnn.pad_packed_sequence(pack_output, batch_first=True)[0]

            batch_emb = []
            for i in range(len(doc_beg_list) - 1):
                doc_beg, doc_end = doc_beg_list[i], doc_beg_list[i + 1]
                base = 0
                for sent_idx in range(doc_end - doc_beg):
                    length = ids_length[doc_beg: doc_end][sent_idx]
                    batch_emb.append(rnn_batch_emb[i, base: base + length])
                    base += length
            batch_emb = pad_sequence_left(batch_emb, batch_first=True, padding_len=self.config['max_tokens_length'])

        reducer = None
        if self.config['pooling'] == 'AWA':
            reducer = self.sent_ids_reducer
        pooling_emb = self.pooling(batch_emb, attention_mask=attention_mask.unsqueeze(dim=-1), dim=1, att_reducer=reducer)
        if self.config['use_pos_emb']:
            pooling_emb = self.sent_pos_encoder(pooling_emb, sent_pos_ids)

        ner_loss = 0
        if train_flag:
            ner_label = torch.tensor(ner_label, device=device, dtype=torch.long)
            if self.config['use_crf']:
                attention_mask = attention_mask.to(torch.uint8)
                ner_loss = -self.crf(ner_score, ner_label, mask=attention_mask, reduction='mean')
            else:
                ner_loss = F.cross_entropy(ner_score.view(-1, len(self.config['NER_LABEL_LIST'])), ner_label.view(-1), ignore_index=-1)
        
            if self.config['cut_word_task']:
                cw_label = torch.tensor(cw_label, device=device, dtype=torch.long)
                cw_score = self.cw_labeler(batch_emb)
                cw_loss = F.cross_entropy(cw_score.view(-1, 2), cw_label.view(-1), ignore_index=-1)
                ner_loss += 0.3 * cw_loss
            if self.config['pos_tag_task']:
                pos_label = torch.tensor(pos_label, device=device, dtype=torch.long)
                pos_score = self.pos_tag_labeler(batch_emb)
                pos_loss = F.cross_entropy(pos_score.view(-1, pos_score.shape[-1]), pos_label.view(-1), ignore_index=-1)
                ner_loss += 0.3 * pos_loss
            if self.config['parser_task']:
                parser_label = torch.tensor(parser_label, device=device, dtype=torch.long)
                parser_q = self.parser_query(batch_emb)
                parser_k = self.parser_key(batch_emb)
                parser_score = torch.bmm(parser_q, parser_k.transpose(-1, -2))
                parser_score = parser_score.masked_fill(
                    (1 - attention_mask.to(dtype=torch.uint8)).unsqueeze(1).expand(-1, attention_mask.shape[-1], -1), -1e9)
                parser_loss = F.cross_entropy(parser_score.view(-1, parser_score.shape[-1]), parser_label.view(-1), ignore_index=-1)
                ner_loss += 0.1 * parser_loss
                #print(1)

        if use_gold:
            ner_pred = ner_label
        else:
            if self.config['use_crf']:
                ner_pred = self.crf.decode(ner_score, mask=attention_mask.to(dtype=torch.uint8))
            else:
                ner_pred = torch.argmax(ner_score, dim=-1).tolist()
        
        doc_ids_emb = []
        doc_ner_pred = []
        doc_sent_emb = []
        doc_ner_score = []
        for i in range(len(doc_beg_list) - 1):
            doc_beg, doc_end = doc_beg_list[i], doc_beg_list[i + 1]
            doc_ids_emb.append(batch_emb[doc_beg: doc_end])
            doc_ner_pred.append(ner_pred[doc_beg: doc_end])
            doc_sent_emb.append(pooling_emb[doc_beg: doc_end])
            doc_ner_score.append(ner_score[doc_beg: doc_end].detach().cpu().numpy())

            self.eval_obj['ner_pred'].append(torch.argmax(ner_score[doc_beg: doc_end], dim=-1).tolist())
        return ner_loss, doc_ids_emb, doc_ner_pred, doc_sent_emb, doc_ner_score

    def generate_train_edag(self, doc_span_drange_list, batch_train):
        EVENT_FIELDS =  self.config['EVENT_FIELDS']
        EVENT_TYPES = self.config['EVENT_TYPES']
        EVENT_TYPE2ID = self.config['EVENT_TYPE2ID']
        doc_edag_info = []
        doc_span2span_label = []
        for span_drange_list, ins in zip(doc_span_drange_list, batch_train):
            span2idx = {}
            if span_drange_list is None:
                doc_edag_info.append(None)
                doc_span2span_label.append(None)
                continue
            
            for i, (span, _) in enumerate(span_drange_list):
                span2idx[span] = i
            edag_info = [None for _ in range(len(EVENT_TYPES))] # event_idx2field_idx2pre_path2cur_span_idx_set
            span2span_label = np.zeros((len(span_drange_list), len(span_drange_list)))

            events = ins['events']
            for event in events:
                event_type = event['event_type']
                event_idx = EVENT_TYPE2ID[event_type]
                fields = EVENT_FIELDS[event_type][0]

                if edag_info[event_idx] is None:
                    edag_info[event_idx] = [{} for _ in fields]
                path_record = []
                for field_idx, field in enumerate(fields):
                    gt_span = event.get(field)
                    if gt_span is not None and gt_span:
                        span_idx = span2idx.get(gt_span)
                    else:
                        span_idx = None

                    for pre_span_idx in path_record:
                        if span_idx is None:
                            continue
                        span2span_label[pre_span_idx, span_idx] = 1
                        span2span_label[span_idx, pre_span_idx] = 1

                    if edag_info[event_idx][field_idx].get(tuple(path_record)) is None:
                        edag_info[event_idx][field_idx][tuple(path_record)] = set()
                    edag_info[event_idx][field_idx][tuple(path_record)].add(span_idx)
                    path_record.append(span_idx)
            doc_edag_info.append(edag_info)
            doc_span2span_label.append(span2span_label)
        return doc_edag_info, doc_span2span_label

def collate_label(labels_list, attention_mask, ids_list):
    span_token_drange_list = []
    for sent_idx, labels in enumerate(labels_list):
        mask = attention_mask[sent_idx]
        ids = ids_list[sent_idx]
        seq_len = len(labels)
        char_s = 0
        while char_s < seq_len:
            if mask[char_s] == 0:
                break
            entity_idx = labels[char_s]
            if entity_idx % 2 == 1:
                char_e = char_s + 1
                while char_e < seq_len and mask[char_e] == 1 and labels[char_e] == entity_idx + 1:
                    char_e += 1

                token_tup = tuple(ids[char_s:char_e])
                drange = (sent_idx, char_s, char_e)

                span_token_drange_list.append((token_tup, drange, entity_idx))

                char_s = char_e
            else:
                char_s += 1
    span_token_drange_list.sort(key=lambda x: x[-1])  # sorted by drange = (sent_idx, char_s, char_e)
    token_tup2dranges = collections.OrderedDict()
    for token_tup, drange, entity_idx in span_token_drange_list:
        # print(tokenizer.decode(token_tup), NER_LABEL_LIST[entity_idx])
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append((drange, entity_idx))
    return token_tup2dranges