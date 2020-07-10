from event_edag_meta import EVENT_TYPES, EVENT_FIELDS, NER_LABEL2ID, NER_LABEL_LIST, EVENT_TYPE_FIELDS_PAIRS, EVENT_TYPE2ID
from utils import strQ2B, sub_list_index, measure_event_table_filling
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
from pyltp import Segmentor

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
        else:
            raise Exception('Not support other basic encoder')
        self.latest_epoch = 0

        if self.config['cut_word_task']:
            cws_model_path = os.path.join(self.config['ltp_path'], 'cws.model')
            segmentor = Segmentor()
            segmentor.load(cws_model_path)
            self.segmentor = segmentor

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
            #self.dev = dev[:self.config['debug_data_num']]
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
            if self.config['use_bert']:
                UNK_ID = self.tokenizer.vocab['[UNK]']
                PAD_ID = self.tokenizer.vocab['[PAD]']
            elif self.config['use_xlnet']:
                UNK_ID = self.tokenizer.convert_tokens_to_ids('<unk>')
                PAD_ID = self.tokenizer.convert_tokens_to_ids('<pad>')

            ids_list = []
            ids_length = []
            attention_mask = []
            labels_list = []
            sentences = ins['sentences'][:MAX_SENT_NUM]
            ins['merge_sentences'] = sentences
            for sentence in sentences:
                ids = []
                mask = []
                if TEXT_NORM:
                    sentence_norm = Traditional2Simplified(strQ2B(sentence)).lower()
                    assert len(sentence_norm) == len(sentence)
                    sentence = sentence_norm
                for char in sentence:
                    ids.append(self.tokenizer.convert_tokens_to_ids(char))
                ids_length.append(len(ids))
                mask = [1 for _ in range(ids_length[-1])]

                pad_num = MAX_TOKENS_LENGTH - ids_length[-1]
                ids.extend([PAD_ID for _ in range(pad_num)])
                mask.extend([0 for _ in range(pad_num)])

                labels_list.append([0 for _ in ids])
                ids_list.append(ids)
                attention_mask.append(mask)
                assert len(ids) == len(mask) == len(labels_list[-1])

            for idx, span in enumerate(ins['ann_valid_mspans']):
                dranges = ins['ann_mspan2dranges'].get(span)
                label = ins['ann_mspan2guess_field'].get(span)
                assert label is not None and dranges is not None
                if label == 'OtherType':
                    continue
                for sent_idx, beg, end in dranges:
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
                len(ids) == len(mask) == len(labels_list[-1])

            ins['ids_list'] = ids_list
            ins['labels_list'] = labels_list
            ins['attention_mask'] = attention_mask
            ins['event_cls'] = event_cls
            ins['ids_length'] = ids_length

            lengths.extend(ids_length)
            #ins['cw_labels_list'] = cw_labels_list
            pbar.update()
        random.shuffle(dataset)

    def preprocess_test(self):
        pass

    def preprocess(self):
        #self.config['max_sent_num'] = 30
        self.preprocess_train(self.train)
        os.sys.stdout.flush()

        #self.config['max_sent_num'] = 30
        self.preprocess_train(self.test)
        os.sys.stdout.flush()

        # self.preprocess_train(self.dev)
        # os.sys.stdout.flush()
        print('preprocess finish')
    
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
        TEST_FILE = self.config['save_test_file']
        MODEL_FILE = self.config['model_file']
        #VALIDATE_DOC_FILE = os.path.join(EVAL_SAVE_DIR, self.config['validate_doc_file'])
        pickle.dump(dev, open(self.config['validate_doc_file'], mode='wb'))

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
                eval_json = self.measure_dee_prediction(total_decode_res, dev)
                print(eval_json[-1]['MicroF1'])
                json.dump(eval_json, open(EVAL_JSON_FILE % epoch, mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)
                pickle.dump(self.model.eval_obj, open(EVAL_OBJ_FILE % epoch, mode='wb'))

            #self.eval_save_test(TEST_FILE % epoch)
            self.latest_epoch += 1 # update epoch
    
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

    'epoch': 300,
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

    'max_tokens_length': 128,
    'max_sent_num': 64,
    'sent_batch_size': 5,

    'dev_ratio': 0.05,
    'text_norm': True,
    
    'use_bert': True,
    'bert_model_name': 'hfl/rbt3',
    'bert_dir': 'rbt3',
    'num_bert_layer': None,
    'bert_add_cls_sep': False,

    'use_xlnet': False,
    'xlnet_doc_bidirection': False,
    'xlnet_doc_reverse': False,
    'xlnet_dir': 'xlnet_chinese',
    'num_xlnet_layer': 4,
    'xlnet_mem_len': 1024,

    'hidden_size': 768,
    'dropout': 0.1,

    'ee_method': 'EDAG', # [GreedyDec, EDAG]
    'use_edag_graph': False, # useless bullshit
    'trainable_pos_emb': False,
    'span_drange_fuse': 'add', # ['add', 'concat']
    'ner_label_count_limit': None,
    'ner_label_sentence_length': 500,

    'cuda': True,
    'use_crf': False,
    'use_token_role': True,
    'use_pos_emb': False,
    'use_doc_enc': False, # consider
    'use_rnn_enc': None, # ['LSTM', 'GRU', None]
    'rnn_bidirection': False,

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
    'cut_word_task': False,
    'validate_doc_file': 'validate_doc.pkl',
    'test_doc_file': 'test_doc.pkl',

    'multilabel_loss': 'multilabel_crossentropy' #['binary', 'multilabel_crossentropy']
}

if __name__ == '__main__':
    torch.cuda.set_device(1)
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