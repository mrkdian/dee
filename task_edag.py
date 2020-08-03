from event_edag_meta import EVENT_TYPES, EVENT_FIELDS, NER_LABEL2ID, NER_LABEL_LIST, EVENT_TYPE_FIELDS_PAIRS, EVENT_TYPE2ID
from utils import strQ2B, sub_list_index, measure_event_table_filling, collate_label
from langconv import Traditional2Simplified
import random
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import XLNetModel, XLNetTokenizer, XLNetConfig
import math
from tqdm import tqdm
import json
import re
import os
import pickle
import time
from model import DocEE
from pyltp import Segmentor, Postagger, Parser
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

POS_TAG_LIST = ['n', 'v', 'wp', 'm', 'other', 'noword']
POS_TAG2ID = { 'noword': 0, 'n': 1, 'v': 2, 'wp': 3, 'm': 4, 'other': 5 }

class DocEETask:
    def __init__(self, config):
        self.config = config
        random_seed = config['random_seed']
        random.seed(random_seed)
        torch.manual_seed(random_seed) # cpu
        torch.cuda.manual_seed(random_seed) #gpu
        np.random.seed(random_seed) #numpy

        if self.config['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_model_name'], cache_dir=config['bert_dir'])
        elif self.config['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained('hfl/chinese-xlnet-base', cache_dir=config['xlnet_dir'])
        elif self.config['use_transformer'] or self.config['use_rnn_basic_encoder']:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=config['bert_dir'])
        else:
            raise Exception('Not support other basic encoder')
        self.latest_epoch = 0

        if self.config['cut_word_task'] or self.config['pos_tag_task'] or self.config['parser_task']:
            cws_model_path = os.path.join(self.config['ltp_path'], 'cws.model')
            segmentor = Segmentor()
            segmentor.load(cws_model_path)
            self.segmentor = segmentor
        if self.config['pos_tag_task'] or self.config['parser_task']:
            pos_model_path = os.path.join(self.config['ltp_path'], 'pos.model')
            postagger = Postagger()
            postagger.load(pos_model_path)
            self.postagger = postagger
        if self.config['parser_task']:
            parser_model_path = os.path.join(self.config['ltp_path'], 'parser.model')
            parser = Parser()
            parser.load(parser_model_path)
            self.parser = parser


    def load_data(self):
        train = json.load(open(self.config['train_file'], mode='r', encoding='utf-8'))
        train = list(map(lambda x: x[1], train))

        test = json.load(open(self.config['test_file'], mode='r', encoding='utf-8'))
        test = list(map(lambda x: x[1], test))

        # dev = json.load(open(self.config['dev_file'], mode='r', encoding='utf-8'))
        # dev = list(map(lambda x: x[1], dev))

        if self.config['debug_data_num'] is not None:
            self.train = train[:self.config['debug_data_num']]
            self.test = test[:self.config['debug_data_num']]
            # self.dev = dev[:self.config['debug_data_num']]
        else:
            self.train = train
            self.test = test
            #self.dev = dev

        self.extend_label()
        print('load data finsih')
    
    def extend_label(self):
        extend_labels = ['StockCode', 'StockAbbr']
        for label in extend_labels:
            NER_LABEL_LIST.append('B-' + label)
            NER_LABEL2ID[NER_LABEL_LIST[-1]] = len(NER_LABEL_LIST) - 1
            NER_LABEL_LIST.append('I-' + label)
            NER_LABEL2ID[NER_LABEL_LIST[-1]] = len(NER_LABEL_LIST) - 1

        # for ins_id, ins in self.train + self.test + self.dev:
        #     for span, label in ins['ann_mspan2guess_field'].items():
        #         if 'B-' + label not in NER_LABEL_LIST:
        #             NER_LABEL_LIST.append('B-' + label)
        #             NER_LABEL2ID[NER_LABEL_LIST[-1]] = len(NER_LABEL_LIST) - 1
        #             NER_LABEL_LIST.append('I-' + label)
        #             NER_LABEL2ID[NER_LABEL_LIST[-1]] = len(NER_LABEL_LIST) - 1
            
    def preprocess_train(self, dataset):
        TEXT_NORM = self.config['text_norm']
        MAX_TOKENS_LENGTH = self.config['max_tokens_length']
        MAX_SENT_NUM = self.config['max_sent_num']
        sentence_sum , sentence_length_sum, truncate_span, span_sum = 0, 0, 0, 0
        pbar = tqdm(total=len(dataset))
        lengths = []
        for _, ins in enumerate(dataset):
            if self.config['use_bert'] or self.config['use_transformer'] or self.config['use_rnn_basic_encoder']:
                UNK_ID = self.tokenizer.vocab['[UNK]']
                PAD_ID = self.tokenizer.vocab['[PAD]']
            elif self.config['use_xlnet']:
                UNK_ID = self.tokenizer.convert_tokens_to_ids('<unk>')
                PAD_ID = self.tokenizer.convert_tokens_to_ids('<pad>')

            ids_list = []
            ids_length = []
            attention_mask = []
            labels_list = []
            cut_word_labels_list = []
            pos_tag_labels_list = []
            parser_labels_list = []

            sentences = ins['sentences'][:MAX_SENT_NUM]
            ins['merge_sentences'] = sentences
            for sentence in sentences:
                ids = []
                mask = []
                labels = []
                cut_word_labels = []
                pos_tag_labels = []
                parser_labels = []

                if TEXT_NORM:
                    sentence_norm = Traditional2Simplified(strQ2B(sentence)).lower()
                    assert len(sentence_norm) == len(sentence)
                    sentence = sentence_norm
                for char in sentence:
                    ids.append(self.tokenizer.convert_tokens_to_ids(char))
                ids_length.append(len(ids))
                mask = [1 for _ in range(ids_length[-1])]
                labels = [0 for _ in range(ids_length[-1])]
                #pos_tag_labels = [0 for _ in range(ids_length[-1])]
                #cut_word_labels = [0 for _ in range(ids_length[-1])]

                pad_num = MAX_TOKENS_LENGTH - ids_length[-1]
                ids.extend([PAD_ID for _ in range(pad_num)])
                mask.extend([0 for _ in range(pad_num)])
                labels.extend([-1 for _ in range(pad_num)])
                #pos_tag_labels.extend([-1 for _ in range(pad_num)])
                #cut_word_labels.extend([-1 for _ in range(pad_num)])

                words = None
                if self.config['cut_word_task']:
                    words = list(self.segmentor.segment(sentence))
                    #words = list(self.segmentor.segment(re.sub('\s', '#', sentence)))
                    for word in words:
                        cut_word_labels.append(1)
                        for _ in word[1:]:
                            cut_word_labels.append(0)
                    assert len(cut_word_labels) == ids_length[-1]
                    cut_word_labels.extend([-1 for _ in range(pad_num)])
                
                postags = None
                if self.config['pos_tag_task']:
                    if words is None:
                        words = list(self.segmentor.segment(sentence))
                    postags = list(self.postagger.postag(words))
                    assert len(postags) == len(words)
                    for idx, word in enumerate(words):
                        if postags[idx].startswith('n'):
                            postags[idx] = 'n'

                        postag_id = POS_TAG2ID.get(postags[idx])
                        if postag_id is None:
                            pos_tag_labels.append(5)
                        else:
                            pos_tag_labels.append(postag_id)

                        for _ in word[1:]:
                            pos_tag_labels.append(0)

                    assert len(pos_tag_labels) == ids_length[-1]
                    pos_tag_labels.extend([-1 for _ in range(pad_num)])
                
                if self.config['parser_task']:
                    if words is None:
                        words = list(self.segmentor.segment(sentence))
                    if postags is None:
                        postags = list(self.postagger.postag(words))
                    arcs = list(self.parser.parse(words, postags))
                    for idx, word in enumerate(words):
                        arc_head = len(''.join(words[:arcs[idx].head - 1]))
                        if arcs[idx].head == 0 or arc_head >= MAX_TOKENS_LENGTH:
                            parser_labels.extend([-1 for _ in word])
                            continue
                        parser_labels.append(arc_head)
                        for _ in word[1:]:
                            parser_labels.append(-1)
                    assert len(parser_labels) == ids_length[-1]
                    parser_labels.extend([-1 for _ in range(pad_num)])

                ids_list.append(ids)
                attention_mask.append(mask)
                labels_list.append(labels)
                cut_word_labels_list.append(cut_word_labels)
                pos_tag_labels_list.append(pos_tag_labels)
                parser_labels_list.append(parser_labels)
                assert len(ids) == len(mask) == len(labels_list[-1])

            for idx, span in enumerate(ins['ann_valid_mspans']):
                dranges = ins['ann_mspan2dranges'].get(span)
                label = ins['ann_mspan2guess_field'].get(span)
                assert label is not None and dranges is not None
                if label == 'OtherType':
                    continue
                for sent_idx, beg, end in dranges:
                    if sent_idx >= MAX_SENT_NUM:
                        continue
                    labels_list[sent_idx][beg] = NER_LABEL2ID['B-' + label]
                    for k in range(beg + 1, end):
                        labels_list[sent_idx][k] = NER_LABEL2ID['I-' + label]
            events = []
            event_cls = [0 for _ in EVENT_TYPES]
            for _, event_type, event in ins['recguid_eventname_eventdict_list']:
                event['event_type'] = event_type
                events.append(event)
                event_cls[EVENT_TYPE2ID.get(event_type)] = 1
            ins['events'] = events

            assert len(ids_list) == len(labels_list) == len(attention_mask)
            for idx in range(len(ids_list)):
                ids_list[idx] = ids_list[idx][:MAX_TOKENS_LENGTH]
                labels_list[idx] = labels_list[idx][:MAX_TOKENS_LENGTH]
                attention_mask[idx] = attention_mask[idx][:MAX_TOKENS_LENGTH]
                ids_length[idx] = MAX_TOKENS_LENGTH if ids_length[idx] > MAX_TOKENS_LENGTH else ids_length[idx]
                #assert len(ids_list[idx]) == len(mask) == len(labels_list[-1])

                cut_word_labels_list[idx] = cut_word_labels_list[idx][:MAX_TOKENS_LENGTH]
                pos_tag_labels_list[idx] = pos_tag_labels_list[idx][:MAX_TOKENS_LENGTH]
                parser_labels_list[idx] = parser_labels_list[idx][:MAX_TOKENS_LENGTH]

            ins['ids_list'] = ids_list
            ins['labels_list'] = labels_list
            ins['attention_mask'] = attention_mask
            ins['event_cls'] = event_cls
            ins['ids_length'] = ids_length
            lengths.extend(ids_length)

            ins['cw_labels_list'] = cut_word_labels_list
            ins['pos_tag_labels_list'] = pos_tag_labels_list
            ins['parser_labels_list'] = parser_labels_list
            pbar.update()
        pbar.close()
        random.shuffle(dataset)

    def preprocess_test(self):
        pass

    def preprocess(self):
        if self.config['remerge_sentence_tokens_length']:
            self.remerge_sentence()

        #self.config['max_sent_num'] = 30
        self.preprocess_train(self.train)
        os.sys.stdout.flush()

        self.config['max_sent_num'] = 64
        self.preprocess_train(self.test)
        os.sys.stdout.flush()

        # self.preprocess_train(self.dev)
        # os.sys.stdout.flush()
        print('preprocess finish')
    
    def remerge_sentence(self):
        remerge_length = self.config['remerge_sentence_tokens_length']
        sent_idx2merge_sent_idx = {}
        for ins in self.train + self.test:
            sentences = ins['sentences']
            span2dranges = ins['ann_mspan2dranges']
            merge_sentences = []
            merge_span2dranges = {}
            cur_sentence = ''
            for sent_idx, sentence in enumerate(sentences):
                if len(sentence + cur_sentence) < remerge_length:
                    cur_beg = len(cur_sentence)
                    cur_sentence += sentence
                else:
                    merge_sentences.append(cur_sentence)
                    cur_beg = 0
                    cur_sentence = sentence

                for span, dranges in span2dranges.items():
                    for span_sent_idx, beg, end in dranges:
                        if span_sent_idx == sent_idx:
                            merge_dranges = merge_span2dranges.get(span)
                            if merge_dranges is not None:
                                merge_dranges.append((len(merge_sentences), cur_beg + beg, cur_beg + end))
                            else:
                                merge_span2dranges[span] = [(len(merge_sentences), cur_beg + beg, cur_beg + end)]

            if len(cur_sentence) > 0:
                merge_sentences.append(cur_sentence)

            for span, dranges in merge_span2dranges.items():
                assert len(dranges) == len(span2dranges[span])
                for sent_idx, beg, end in dranges:
                    assert merge_sentences[sent_idx][beg: end] == span
            
            ins['sentences'] = merge_sentences
            ins['ann_mspan2dranges'] = merge_span2dranges
    
    def init_model(self):
        basic_encoder = None
        if self.config['use_bert']:
            bert_config = BertConfig.from_pretrained(self.config['bert_model_name'], cache_dir=self.config['bert_dir'])
            if self.config['num_bert_layer'] is not None:
                bert_config.num_hidden_layers = self.config['num_bert_layer']
            bert = BertModel.from_pretrained(self.config['bert_model_name'], cache_dir=self.config['bert_dir'], config=bert_config)
            basic_encoder = bert
        elif self.config['use_xlnet']:
            xlnet_config = XLNetConfig.from_pretrained('hfl/chinese-xlnet-base', cache_dir=self.config['xlnet_dir'])
            xlnet_config.n_layer = self.config['num_xlnet_layer']
            xlnet_config.mem_len = self.config['xlnet_mem_len']
            xlnet = XLNetModel.from_pretrained('hfl/chinese-xlnet-base', cache_dir=self.config['xlnet_dir'], config=xlnet_config)
            basic_encoder = xlnet
        elif self.config['use_transformer']:
            bert_config = BertConfig.from_pretrained('bert-base-chinese', cache_dir=self.config['bert_dir'])
            if self.config['num_transformer_layer'] is not None:
                bert_config.num_hidden_layers = self.config['num_transformer_layer']
            transf = BertModel(bert_config)
            basic_encoder = transf
        elif self.config['use_rnn_basic_encoder']:
            pass
        else:
            raise Exception('Not support other basic encoder')

        self.model = DocEE(self.config, basic_encoder, self.tokenizer)
        if self.config['cuda']:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        if self.config['resume_model']:
            OUTPUT_DIR = self.config['output_dir']
            MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, self.config['model_save_dir'])
            if os.path.exists(MODEL_SAVE_DIR):
                cpt_file_names = os.listdir(MODEL_SAVE_DIR)
                if len(cpt_file_names) > 0:
                    epoch_record = []
                    for cpt_file_name in cpt_file_names:
                        epoch_record.append(int(cpt_file_name.split('-')[-1].split('.')[0]))
                    epoch_record.sort()
                    latest_epoch = epoch_record[-1]
                    self.latest_epoch = latest_epoch + 1

                    latest_model_file_name = os.path.join(MODEL_SAVE_DIR, self.config['model_file'] % (self.config['ee_method'], latest_epoch))
                    if self.config['cuda']:
                        store_dict = torch.load(latest_model_file_name, map_location=torch.device('cuda'))
                    else:
                        store_dict = torch.load(latest_model_file_name, map_location='cpu')
                    self.model.load_state_dict(store_dict['model_state'])
                    self.optimizer.load_state_dict(store_dict['optimizer_state'])
                    print('resume train from %s' % latest_model_file_name)
        print('model init finish')

    def get_teacher_prob(self):
        if self.latest_epoch <= self.config['schedule_epoch_start']:
            return 1
        elif self.latest_epoch <= self.config['schedule_epoch_start'] + self.config['schedule_epoch_length']:
            left_prob = 1 - self.config['min_teacher_prob']
            left_ratio = 1 - (self.latest_epoch - self.config['schedule_epoch_start']) / self.config['schedule_epoch_length']
            return self.config['min_teacher_prob'] + left_prob * left_ratio
        else:
            return self.config['min_teacher_prob']


    def train_eval_by_epoch(self):
        EPOCH = self.config['epoch']
        DOC_BATCH_SIZE = self.config['train_doc_batch_size']
        EVAL_DOC_BATCH_SIZE = self.config['eval_doc_batch_size']

        DEV_REAIO = self.config['dev_ratio']
        dev_num = math.ceil(len(self.train) * DEV_REAIO)
        dev = self.test
        part_train = self.train

        BATCH_NUM = math.ceil(len(part_train) / DOC_BATCH_SIZE)
        EVAL_BATCH_NUM = math.ceil(len(dev) / EVAL_DOC_BATCH_SIZE)

        OUTPUT_DIR = self.config['output_dir']
        EVAL_SAVE_DIR = os.path.join(OUTPUT_DIR, self.config['eval_save_dir'])

        EVAL_JSON_FILE = os.path.join(EVAL_SAVE_DIR, self.config['eval_json_file'])
        EVAL_OBJ_FILE = os.path.join(EVAL_SAVE_DIR, self.config['eval_obj_file'])
        EVAL_NER_FILE = os.path.join(EVAL_SAVE_DIR, self.config['eval_ner_json_file'])
        TEST_FILE = self.config['save_test_file']
        MODEL_FILE = self.config['model_file']
        #VALIDATE_DOC_FILE = os.path.join(EVAL_SAVE_DIR, self.config['validate_doc_file'])
        #pickle.dump(dev, open(self.config['validate_doc_file'], mode='wb'))

        if self.config['accum_batch_size'] is not None:
            accum_batch_size = self.config['accum_batch_size']
            accum_step = accum_batch_size // DOC_BATCH_SIZE

        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        if not os.path.exists(EVAL_SAVE_DIR):
            os.mkdir(EVAL_SAVE_DIR)

        for epoch in range(self.latest_epoch, EPOCH):
            torch.cuda.empty_cache()
            os.sys.stdout.flush()
            print('%d-----------------' % epoch)
            if not self.config['skip_train']:
                teacher_prob = 1
                if self.config['use_schedule_sampling']:
                    teacher_prob = self.get_teacher_prob()
                    print('use schedule sampling, teacher_prob =', teacher_prob)

                print_loss = 0
                random.shuffle(part_train)
                self.model.train()
                with tqdm(total=BATCH_NUM) as pbar:
                    for batch_num in range(BATCH_NUM):
                        batch_beg = batch_num * DOC_BATCH_SIZE
                        batch_end = (batch_num + 1) * DOC_BATCH_SIZE
                        batch_train = part_train[batch_beg: batch_end]
                        
                        if random.random() < teacher_prob:
                            use_gold = True
                        else:
                            use_gold = False
                        #optimizer.zero_grad()
                        loss = self.model(batch_train, train_flag=True, dev_flag=False, use_gold=use_gold)[0]

                        if self.config['accum_batch_size'] is not None and epoch > self.config['not_accum_optim_epoch']:
                            loss /= accum_step
                            loss.backward()
                            if (batch_num + 1) % accum_step == 0 or batch_num == BATCH_NUM - 1:
                                self.optimizer.step()
                                self.model.zero_grad()
                        else:
                            loss.backward()
                            self.optimizer.step()
                            self.model.zero_grad()
                        print_loss += loss.cpu().detach().numpy()
                        pbar.set_description('total_loss: %f' % (print_loss / (batch_num + 1)))
                        pbar.update()
                if self.config['save_model']:
                    self.save_model_checkpoint(MODEL_FILE % (self.config['ee_method'], epoch), epoch)
            os.sys.stdout.flush()

            if not self.config['skip_eval']:
                torch.cuda.empty_cache()
                self.model.eval()
                self.model.init_eval_obj()
                total_decode_res = []
                with tqdm(total=EVAL_BATCH_NUM) as pbar:
                    for batch_num in range(EVAL_BATCH_NUM):
                        batch_beg = batch_num * EVAL_DOC_BATCH_SIZE
                        batch_end = (batch_num + 1) * EVAL_DOC_BATCH_SIZE
                        batch_dev = dev[batch_beg: batch_end]

                        with torch.no_grad():
                            doc_decode_res = self.model(batch_dev, train_flag=False, dev_flag=True, use_gold=False)[1]
                        total_decode_res.extend(doc_decode_res)
                        pbar.update()
                os.sys.stdout.flush()
                if not self.config['only_ner']:
                    eval_json = self.measure_dee_prediction(total_decode_res, dev)
                    print('event extreaction f1:', eval_json[-1]['MicroF1'])
                    json.dump(eval_json, open(EVAL_JSON_FILE % epoch, mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)
                    pickle.dump(self.model.eval_obj, open(EVAL_OBJ_FILE % epoch, mode='wb'))

                    ner_res = self.measure_ner_prediction(self.model.eval_obj['ner_gt'], self.model.eval_obj['ner_pred'], dev)
                    json.dump(ner_res, open(EVAL_NER_FILE % epoch, mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)
                    print('ner f1:', ner_res['micro_f1'])
                else:
                    ner_res = self.measure_ner_prediction(self.model.eval_obj['ner_gt'], self.model.eval_obj['ner_pred'], dev)
                    json.dump(ner_res, open(EVAL_NER_FILE % epoch, mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)
                    print('ner f1:', ner_res['micro_f1'])
            #self.eval_save_test(TEST_FILE % epoch)
            self.latest_epoch += 1 # update epoch
    
    def measure_ner_prediction(self, ner_gt, ner_pred, dev):
        NER_LABEL2ID = self.config['NER_LABEL2ID']
        NER_LABEL_LIST = self.config['NER_LABEL_LIST']
        ner_res = {}
        for label in NER_LABEL_LIST[1:]:
            ner_res[label[2:]] = { 'tp': 0, 'fp': 0, 'fn': 0 }

        for doc_ner_gt, doc_ner_pred, ins in zip(ner_gt, ner_pred, dev):
            ids2drange_gt = collate_label(doc_ner_gt, ins['attention_mask'], ins['ids_list'])
            ids2drange_pred = collate_label(doc_ner_pred, ins['attention_mask'], ins['ids_list'])
            for k in ids2drange_gt.keys():
                ids2drange_gt[k] = dict(ids2drange_gt[k])
            for k in ids2drange_pred.keys():
                ids2drange_pred[k] = dict(ids2drange_pred[k])

            for gt_ids, dranges in ids2drange_gt.items():
                pred_dranges = ids2drange_pred.get(gt_ids)
                if pred_dranges is not None:
                    for drange, label in dranges.items():
                        pred_label = pred_dranges.get(drange)
                        if pred_label == label:
                            ner_res[NER_LABEL_LIST[label][2:]]['tp'] += 1
                            pred_dranges.pop(drange)
                        else:
                            if pred_label is not None:
                                ner_res[NER_LABEL_LIST[pred_label][2:]]['fp'] += 1
                                pred_dranges.pop(drange)
                            ner_res[NER_LABEL_LIST[label][2:]]['fn'] += 1
                else:
                    for drange, label in dranges.items():
                        ner_res[NER_LABEL_LIST[label][2:]]['fn'] += 1
            
            for pred_ids, dranges in ids2drange_pred.items():
                for drange, label in dranges.items():
                    ner_res[NER_LABEL_LIST[label][2:]]['fp'] += 1

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_f1 = 0
        for label in NER_LABEL_LIST[1:]:
            label = label[2:]
            tp, fp, fn = ner_res[label]['tp'], ner_res[label]['fp'], ner_res[label]['fn']
            total_tp += tp
            total_fp += fp
            total_fn += fn

            p = 0 if tp == 0 else tp / (tp + fp)
            r = 0 if tp == 0 else tp / (tp + fn)
            f1 = 0 if p + r == 0 else (2 * p * r) / (p + r)
            ner_res[label]['f1'] = f1
            ner_res[label]['precision'] = p
            ner_res[label]['recall'] = r
            total_f1 += f1

        total_p = 0 if total_tp == 0 else total_tp / (total_tp + total_fp)
        total_r = 0 if total_tp == 0 else total_tp / (total_tp + total_fn)
        ner_res['precision'] = total_p
        ner_res['recall'] = total_r
        ner_res['micro_f1'] = 0 if total_p + total_r == 0 else (2 * total_p * total_r) / (total_p + total_r)
        ner_res['macro_f1'] = total_f1 / (len(NER_LABEL_LIST) - 1)
        return ner_res

        

    def save_model_checkpoint(self, cpt_file_name, epoch):
        OUTPUT_DIR = self.config['output_dir']
        MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, self.config['model_save_dir'])
        if not os.path.exists(MODEL_SAVE_DIR):
            os.mkdir(MODEL_SAVE_DIR)
        cpt_file_name = os.path.join(MODEL_SAVE_DIR, cpt_file_name)
        store_dict = {
            'setting': self.config,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(store_dict, cpt_file_name)

    def eval_save_test(self, save_file_name):
        EVENT_TYPES, EVENT_FIELDS, EVENT_TYPE_FIELDS_PAIRS = \
            self.config['EVENT_TYPES'], self.config['EVENT_FIELDS'], self.config['EVENT_TYPE_FIELDS_PAIRS']
        TEST_DOC_BATCH_SIZE = self.config['test_doc_batch_size']
        TEST_BATCH_NUM = math.ceil(len(self.test) / TEST_DOC_BATCH_SIZE)
        TEST_SAVE_DIR = os.path.join(self.config['output_dir'], self.config['test_save_dir'])
        if not os.path.exists(TEST_SAVE_DIR):
            os.mkdir(TEST_SAVE_DIR)
        save_file_name = os.path.join(TEST_SAVE_DIR, save_file_name)

        self.model.eval()
        total_decode_res = []
        with tqdm(total=TEST_BATCH_NUM) as pbar:
            for batch_num in range(TEST_BATCH_NUM):
                batch_beg = batch_num * TEST_DOC_BATCH_SIZE
                batch_end = (batch_num + 1) * TEST_DOC_BATCH_SIZE
                batch_test = self.test[batch_beg: batch_end]

                if self.config['debug_data_id_test']:
                    skip = True
                    for ins in batch_test:
                        if ins['doc_id'] == self.config['debug_data_id_test']:
                            skip = False
                    if skip:
                        continue

                with torch.no_grad():
                    doc_decode_res = self.model(batch_test, train_flag=False, dev_flag=False, use_gold=False)[1]
                total_decode_res.extend(doc_decode_res)
                pbar.update()
        self.decode_drange2text(total_decode_res, self.test)

        for ins, decode_res in zip(self.test, total_decode_res):
            mult_ans = []
            for event_idx, mult_res in enumerate(decode_res):
                if mult_res is None or len(mult_res) < 1:
                    continue
                event_type = EVENT_TYPES[event_idx]
                event_fields = EVENT_FIELDS[event_type][0]
                for res in mult_res:
                    ans = {'event_type': event_type}
                    for field_idx, span in enumerate(res):
                        if span is None:
                            continue
                        ans[event_fields[field_idx]] = span
                    mult_ans.append(ans)
            ins['events'] = mult_ans

        with_content = open(save_file_name + '-with_content', mode='w', encoding='utf-8')
        with open(save_file_name, mode='w', encoding='utf-8') as f:
            for ins in self.test:
                write_obj = {}
                write_obj['doc_id'] = ins['doc_id']
                write_obj['events'] = ins['events']
                f.write(json.dumps(write_obj, ensure_ascii=False) + '\n')
                with_content.write(json.dumps(ins, ensure_ascii=False) + '\n')
        return self.test

    def decode_drange2text(self, doc_decode_res, batch_test):
        for i, decode_res in enumerate(doc_decode_res):
            merge_sentences = batch_test[i]['merge_sentences']
            for event_idx, res in enumerate(decode_res):
                if res is None:
                    continue
                for fields in res:
                    for j, drange in enumerate(fields):
                        if drange is None:
                            continue
                        sent_idx, char_s, char_e = drange
                        fields[j] = merge_sentences[sent_idx][char_s: char_e]

    def measure_dee_prediction(self, doc_decode_res, batch_dev):
        EVENT_TYPES, EVENT_FIELDS, EVENT_TYPE_FIELDS_PAIRS = \
            self.config['EVENT_TYPES'], self.config['EVENT_FIELDS'], self.config['EVENT_TYPE_FIELDS_PAIRS']
        
        self.decode_drange2text(doc_decode_res, batch_dev)
    
        gt_decode_res = []
        for ins in batch_dev:
            decode_res = [None for _ in EVENT_TYPES]
            events = ins['events']
            for event in events:
                event_type = event['event_type']
                event_idx = EVENT_TYPES.index(event_type)
                if decode_res[event_idx] == None:
                    decode_res[event_idx] = []
                event_fields = EVENT_FIELDS[event_type]
                res = []
                for field in event_fields[0]:
                    span = None
                    if field in event and event[field]:
                        span = event[field]
                    res.append(span)
                decode_res[event_idx].append(res)
            gt_decode_res.append(decode_res)
        eval_res = measure_event_table_filling(doc_decode_res, gt_decode_res, EVENT_TYPE_FIELDS_PAIRS, dict_return=True)
        return eval_res

default_task_config = {
    'test_file': 'edag_data/test.json',
    'train_file': 'edag_data/sample_train.json',
    'dev_file': 'edag_data/dev.json',
    'output_dir': 'output_edag',
    'model_save_dir': 'save_model',
    'eval_save_dir': 'save_eval',
    'test_save_dir': 'save_test',
    'eval_json_file': 'eval-%d.json',
    'eval_obj_file': 'eval-obj-%d.pkl',
    'save_test_file': 'test-%d.txt',
    'model_file': '%s-%d.cpt',
    'random_seed': 666,
    'eval_ner_json_file': 'eval-ner-%d.json',

    'epoch': 90,
    'train_doc_batch_size': 2,
    'eval_doc_batch_size': 2,
    'test_doc_batch_size': 2,
    'accum_batch_size': 64,
    'not_accum_optim_epoch': 1,

    'use_schedule_sampling': True,
    'min_teacher_prob': 0.1,
    'schedule_epoch_start': 1,
    'schedule_epoch_length': 10,# the number of epochs to linearly transit to the min_teacher_prob

    'skip_train': False,
    'skip_eval': False,
    'save_model': True,
    'resume_model': True,
    'only_ner': False,

    'max_tokens_length': 128, # 128 for no merge, 500 for merge
    'max_sent_num': 64, # 64 for no merge, 5 for merge
    'sent_batch_size': 5,
    'remerge_sentence_tokens_length': False, # usually 500

    'dev_ratio': 0.05,
    'text_norm': True,
    
    ######## basic encoder ###########
    'use_rnn_basic_encoder': None, #['LSTM', 'GRU', None]
    'basic_rnn_bidirection': True,
    'basic_num_rnn_layer': 4,

    'use_transformer': False,
    'num_transformer_layer': 4,

    'use_bert': True,
    'bert_model_name': 'bert-base-chinese', # ['hfl/rbt3', 'bert-base-chinese']
    'bert_dir': 'bert_base_chinese', # ['rbt3', 'bert_base_chinese']
    'num_bert_layer': 4,
    'bert_add_cls_sep': False,

    'use_xlnet': False,
    'xlnet_doc_bidirection': False,
    'xlnet_doc_reverse': False,
    'xlnet_dir': 'xlnet_chinese',
    'num_xlnet_layer': 4,
    'xlnet_mem_len': 1024,
    ##################################

    'hidden_size': 768,
    'dropout': 0.1,

    'ee_method': 'EDAG', # [GreedyDec, EDAG]
    'use_edag_graph': False, # useless bullshit
    'use_path_mem': False,
    'trainable_pos_emb': False,
    'span_drange_fuse': 'add', # ['add', 'concat']
    'ner_label_count_limit': None,
    'ner_label_sentence_length': 500,

    'cuda': True,

    'use_crf': False,
    'use_token_role': True,
    'use_pos_emb': False,
    'use_doc_enc': True, # consider
    'use_rnn_enc': None, # ['LSTM', 'GRU', None]
    'rnn_bidirection': True,

    'num_tf_layer': 4,
    'ff_size': 1024,
    'neg_field_loss_scaling': 1.0, #prefer FN over FP
    'pooling': 'max', # ['max', 'mean', 'AWA', 'AWA-R']
    'learning_rate': 1e-4,

    'EVENT_TYPES': EVENT_TYPES,
    'EVENT_FIELDS': EVENT_FIELDS,
    'NER_LABEL_LIST': NER_LABEL_LIST,
    'NER_LABEL2ID': NER_LABEL2ID,
    'EVENT_TYPE2ID': EVENT_TYPE2ID,
    'EVENT_TYPE_FIELDS_PAIRS': EVENT_TYPE_FIELDS_PAIRS,

    'debug_data_num': None,
    'debug_data_id_test': None,
    'ltp_path': 'ltp_model',

    'cut_word_task': True,
    'pos_tag_task': True,
    'POS_TAG_LIST': POS_TAG_LIST,
    'POS_TAG2ID': POS_TAG2ID,
    'parser_task': True,


    'validate_doc_file': 'validate_doc.pkl',
    'test_doc_file': 'test_doc.pkl',

    'multilabel_loss': 'binary' #['binary', 'multilabel_crossentropy']
}

if __name__ == '__main__':
    torch.cuda.set_device(0)
    task_config = default_task_config
    for k, v in task_config.items():
        if not isinstance(v, (int, str, float)):
            continue
        print(k + ':', str(v))
    print('-----------------')

    task = DocEETask(task_config)
    task.load_data()
    task.preprocess()
    task.init_model()
    task.train_eval_by_epoch()