import pickle
from task import DocEETask, default_task_config
import os
import json
import torch
from collections import OrderedDict
import math
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from event_meta import NER_LABEL2ID, EVENT_TYPE_FIELDS_PAIRS, EVENT_TYPES, EVENT_FIELDS


def softmax(x):
    """ softmax function """
    x -= np.max(x, axis = 1, keepdims = True)
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

class BlendingTask:
    def __init__(self, config, dee_task):
        self.config = config
        self.train = pickle.load(open(config['train_pkl'], mode='rb'))
        self.test = pickle.load(open(config['test_pkl'], mode='rb'))
        self.dee_task = dee_task
    
    def collate_model(self):
        eval_dir = self.config['ensemble_model_eval_dir']
        f1_res = []
        for file_name in os.listdir(eval_dir):
            if file_name.endswith('json'):
                epoch = int(file_name.split('-')[1].split('.')[0])
                eval_json = json.load(open(os.path.join(eval_dir, file_name), mode='r', encoding='utf-8'))
                f1 = eval_json[-1]['MicroF1']
                f1_res.append((epoch, f1))
        f1_res = sorted(f1_res, key=lambda x: x[1], reverse=True)
        if self.config['ensemble_best_epochs'] is not None:
            f1_res = f1_res[:self.config['ensemble_best_epochs']]
        
        collate_info = OrderedDict()
        collate_epoch = list(map(lambda x: x[0], f1_res))
        model_dir = self.config['ensemble_model_dir']
        pbar = tqdm(total=len(f1_res))
        for model_file_name in os.listdir(model_dir):
            epoch = int(model_file_name.split('-')[1].split('.')[0])
            if epoch in collate_epoch:
                model_file_name = os.path.join(model_dir, model_file_name)
                store_dict = torch.load(model_file_name, map_location='cpu')
                collate_info[epoch] = {
                    'model_state_dict': store_dict['model_state']
                }
                pbar.update()
        self.collate_info = collate_info
        print('collate model fin')
        
    def init_base_model(self):
        self.dee_task.config['resume_model'] = False
        self.dee_task.init_model()
        self.model = self.dee_task.model

    def fit_ner(self, dataset):
        pbar = tqdm(total=len(self.collate_info))
        for key, info in self.collate_info.items():
            self.model.load_state_dict(info['model_state_dict'])
            self.model.eval()
            info['ner_score'] = []
            info['ner_target'] = []
            TEST_DOC_BATCH_SIZE = self.dee_task.config['test_doc_batch_size']
            TEST_BATCH_NUM = math.ceil(len(dataset) / TEST_DOC_BATCH_SIZE)
            for batch_num in range(TEST_BATCH_NUM):
                batch_beg = batch_num * TEST_DOC_BATCH_SIZE
                batch_end = (batch_num + 1) * TEST_DOC_BATCH_SIZE
                batch_test = dataset[batch_beg: batch_end]
                with torch.no_grad():
                    ner_score = self.model.do_ner(batch_test, train_flag=False, use_gold=False)[-1]
                for idx, ins in enumerate(batch_test):
                    for sent_idx, l in enumerate(ins['ids_length']):
                        info['ner_score'].append(softmax(ner_score[idx][sent_idx, :l]))
                        info['ner_target'].append(ins['labels_list'][sent_idx][: l])
            info['ner_score'] = np.vstack(info['ner_score'])
            info['ner_target'] = np.hstack(info['ner_target'])
            pbar.update()
        print('generate ner feature fin and begin to fit')

        ner_feature = []
        ner_target = None
        for key, info in self.collate_info.items():
            ner_feature.append(info['ner_score'])
            if ner_target is None:
                ner_target = info['ner_target']
            assert np.sum(ner_target != info['ner_target']) == 0
        ner_feature = np.hstack(ner_feature)
        self.ner_feature = ner_feature

        if self.config['debug_data_num'] is not None:
            ner_feature = ner_feature[:self.config['debug_data_num']]
            ner_target = ner_target[:self.config['debug_data_num']]
        ner_clf = LogisticRegression(
            random_state=self.config['random_seed'],
            multi_class='multinomial', 
            verbose=True, max_iter=2000, 
            n_jobs=-1
        )
        print(ner_feature.shape, ner_target.shape)
        ner_clf.fit(ner_feature, ner_target)
        self.ner_clf = ner_clf
        pickle.dump(ner_clf, open('blending_ner_model.pkl', mode='wb'))
        return ner_feature

    def fit_edag(self, dataset):
        TEST_DOC_BATCH_SIZE = self.dee_task.config['test_doc_batch_size']
        TEST_BATCH_NUM = math.ceil(len(dataset) / TEST_DOC_BATCH_SIZE)

        event2field2clf = {}
        for event_idx, (event_type, fields) in enumerate(EVENT_TYPE_FIELDS_PAIRS):
            event2field2clf[event_idx] = []
            for field_idx, field in enumerate(fields):
                event2field2clf[event_idx].append({
                    'feature': [],
                    'target': [],
                    'clf': LogisticRegression(random_state=self.config['random_seed'])
                })
        event_clf_feature = []
        event_clf_target = []
        event_clf = MLPClassifier(hidden_layer_sizes=(9,), max_iter=8000)

        with torch.no_grad():
            pbar = tqdm(total=TEST_BATCH_NUM)
            ner_feature = []
            for batch_num in range(TEST_BATCH_NUM):
                batch_beg = batch_num * TEST_DOC_BATCH_SIZE
                batch_end = (batch_num + 1) * TEST_DOC_BATCH_SIZE
                batch_test = dataset[batch_beg: batch_end]
                for key, info in self.collate_info.items():
                    self.model.load_state_dict(info['model_state_dict'])
                    self.model.eval()
                    info['ner_score'] = []
                    ner_loss, doc_ids_emb, doc_ner_pred, doc_sent_emb, ner_score = \
                        self.model.do_ner(batch_test, train_flag=False, use_gold=False)
                    info['doc_ids_emb'] = doc_ids_emb
                    info['doc_sent_emb'] = doc_sent_emb
                    info['doc_ner_pred'] = doc_ner_pred
                    for idx, ins in enumerate(batch_test):
                        for sent_idx, l in enumerate(ins['ids_length']):
                            info['ner_score'].append(softmax(ner_score[idx][sent_idx, :l]))
                    info['ner_score'] = np.vstack(info['ner_score'])

                ner_feature = []
                for key, info in self.collate_info.items():
                    ner_feature.append(info['ner_score'])
                ner_feature = np.hstack(ner_feature)
                ner_pred_array = self.ner_clf.predict(ner_feature)

                doc_ner_pred = []
                doc_ner_gt = []
                base = 0
                for idx, ins in enumerate(batch_test):
                    ner_pred_list = []
                    for sent_idx, l in enumerate(ins['ids_length']):
                        ner_pred_list.append(ner_pred_array[base: base + l].tolist())
                        base += l
                    doc_ner_pred.append(ner_pred_list)
                    doc_ner_gt.append(ins['labels_list'])

                model_score_info = []
                for key, info in self.collate_info.items():
                    self.model.load_state_dict(info['model_state_dict'])
                    self.model.eval()
                    doc_ids_emb = info['doc_ids_emb']
                    doc_sent_emb = info['doc_sent_emb']
                    #pred
                    doc_span_drange_list, doc_span_emb, doc_sent_emb = self.model.pooling_for_span(
                        doc_ids_emb, doc_ner_pred, doc_sent_emb, batch_test, train_flag=False
                    )
                    doc_edag_info, doc_span2span_info = self.model.generate_train_edag(doc_span_drange_list, batch_test)
                    doc_score_info = self.model.edag_dec(
                        doc_span_drange_list, doc_span_emb, doc_sent_emb, doc_edag_info,
                        batch_test, train_flag=True, use_gold=False, doc_span2span_info=doc_span2span_info
                    )[2]

                    #gt
                    doc_span_drange_list, doc_span_emb, doc_sent_emb = self.model.pooling_for_span(
                        doc_ids_emb, doc_ner_gt, doc_sent_emb, batch_test, train_flag=False
                    )
                    doc_edag_info, doc_span2span_info = self.model.generate_train_edag(doc_span_drange_list, batch_test)
                    doc_gt_score_info = self.model.edag_dec(
                        doc_span_drange_list, doc_span_emb, doc_sent_emb, doc_edag_info,
                        batch_test, train_flag=True, use_gold=False, doc_span2span_info=doc_span2span_info
                    )[2]
                    doc_score_info.extend(doc_gt_score_info)
                    info['doc_score_info'] = doc_score_info
                
                doc2event2prev_path2score = {}
                doc2event_score = {}
                for key, info in self.collate_info.items():
                    doc_score_info = info['doc_score_info']
                    for batch_idx, score_info in enumerate(doc_score_info):
                        if batch_idx not in doc2event2prev_path2score:
                            doc2event2prev_path2score[batch_idx] = {}
                        if batch_idx not in doc2event_score:
                            doc2event_score[batch_idx] = {'score': [], 'target': []}
                        doc2event_score[batch_idx]['score'].append(score_info['event_type_score'])
                        doc2event_score[batch_idx]['target'].append(score_info['event_type_target'])

                        for event_idx, edag_score_info in enumerate(score_info['edag_score_info']):
                            if len(edag_score_info) < 1:
                                continue
                            if event_idx not in doc2event2prev_path2score[batch_idx]:
                                doc2event2prev_path2score[batch_idx][event_idx] = {}
                            for prev_path, path_score in edag_score_info.items():
                                if prev_path not in doc2event2prev_path2score[batch_idx][event_idx]:
                                    doc2event2prev_path2score[batch_idx][event_idx][prev_path] = {
                                    'score': [],
                                    'target': [],
                                    'field_idx': path_score['field_idx']
                                }
                                assert event_idx == path_score['event_idx']
                                assert doc2event2prev_path2score[batch_idx][event_idx][prev_path]['field_idx'] == path_score['field_idx']
                                doc2event2prev_path2score[batch_idx][event_idx][prev_path]['score'].append(path_score['logp_score'])
                                doc2event2prev_path2score[batch_idx][event_idx][prev_path]['target'].append(path_score['target'])

                for batch_idx, event2prev_path2score in doc2event2prev_path2score.items():
                    for event_idx, prev_path2score in event2prev_path2score.items():
                        for prev_path, path_score in prev_path2score.items():
                            field_idx = path_score['field_idx']
                            score_feature = np.hstack(path_score['score'])
                            score_target = np.array(path_score['target'][0])
                            event2field2clf[event_idx][field_idx]['feature'].append(score_feature)
                            event2field2clf[event_idx][field_idx]['target'].append(score_target)

                for batch_idx, event_score in doc2event_score.items():
                    event_clf_feature.append(np.hstack(event_score['score']))
                    event_clf_target.append(np.array(event_score['target'][0]))
                pbar.update()
        for event_idx, (event_type, fields) in enumerate(EVENT_TYPE_FIELDS_PAIRS):
            for field_idx, field in enumerate(fields):
                field_feature = np.vstack(event2field2clf[event_idx][field_idx]['feature'])
                field_target = np.hstack(event2field2clf[event_idx][field_idx]['target'])
                if np.sum(field_target) == 0:
                    event2field2clf[event_idx][field_idx]['clf'] = None
                else:
                    event2field2clf[event_idx][field_idx]['clf'].fit(field_feature, field_target)
                #print(event_type, field, field_feature.shape[0])
        
        event_clf_feature = np.vstack(event_clf_feature)
        event_clf_target = np.vstack(event_clf_target)
        event_clf.fit(event_clf_feature, event_clf_target)
        
        self.event2field2clf = event2field2clf
        self.event_clf = event_clf
        pickle.dump(event2field2clf, open('blending_edag_field_model.pkl', mode='wb'))
        pickle.dump(event_clf, open('blending_edag_event_model.pkl', mode='wb'))

    def edag_predict(self):
        dataset = self.test
        TEST_DOC_BATCH_SIZE = self.dee_task.config['test_doc_batch_size']
        TEST_BATCH_NUM = math.ceil(len(dataset) / TEST_DOC_BATCH_SIZE)

        with torch.no_grad():
            pbar = tqdm(total=TEST_BATCH_NUM)
            ner_feature = []
            total_decode_res = []
            for batch_num in range(TEST_BATCH_NUM):
                batch_beg = batch_num * TEST_DOC_BATCH_SIZE
                batch_end = (batch_num + 1) * TEST_DOC_BATCH_SIZE
                batch_test = dataset[batch_beg: batch_end]

                #do ner
                for key, info in self.collate_info.items():
                    self.model.load_state_dict(info['model_state_dict'])
                    self.model.eval()
                    info['ner_score'] = []
                    ner_loss, doc_ids_emb, doc_ner_pred, doc_sent_emb, ner_score = \
                        self.model.do_ner(batch_test, train_flag=False, use_gold=False)
                    info['doc_ids_emb'] = doc_ids_emb
                    info['doc_sent_emb'] = doc_sent_emb
                    for idx, ins in enumerate(batch_test):
                        for sent_idx, l in enumerate(ins['ids_length']):
                            info['ner_score'].append(softmax(ner_score[idx][sent_idx, :l]))
                    info['ner_score'] = np.vstack(info['ner_score'])

                ner_feature = []
                for key, info in self.collate_info.items():
                    ner_feature.append(info['ner_score'])
                ner_feature = np.hstack(ner_feature)
                ner_pred_array = self.ner_clf.predict(ner_feature)

                doc_ner_pred = []
                base = 0
                for idx, ins in enumerate(batch_test):
                    ner_pred_list = []
                    for sent_idx, l in enumerate(ins['ids_length']):
                        ner_pred_list.append(ner_pred_array[base: base + l].tolist())
                        base += l
                    doc_ner_pred.append(ner_pred_list)

                #event cls
                doc_event_type_score = []
                for key, info in self.collate_info.items():
                    self.model.load_state_dict(info['model_state_dict'])
                    self.model.eval()
                    doc_ids_emb = info['doc_ids_emb']
                    doc_sent_emb = info['doc_sent_emb']
                    doc_span_drange_list, doc_span_emb, doc_sent_emb = self.model.pooling_for_span(
                        doc_ids_emb, doc_ner_pred, doc_sent_emb, batch_test, train_flag=False
                    )
                    info['doc_span_drange_list'] = doc_span_drange_list
                    info['doc_span_emb'] = doc_span_emb
                    info['doc_sent_emb'] = doc_sent_emb
                    info['event_type_score'] = []

                    for sent_emb in doc_sent_emb:
                        event_type_score = []
                        for event_table in self.model.event_tables:
                                event_type_score.append(event_table(sent_context_emb=sent_emb)) # already log softmax
                        event_type_score = torch.cat(event_type_score, dim=0)
                        info['event_type_score'].append(np.exp(event_type_score.detach().cpu().numpy()).reshape(-1))
                    doc_event_type_score.append(np.vstack(info['event_type_score']))
                doc_event_type_score = np.hstack(doc_event_type_score)
                doc_event_type_pred = self.event_clf.predict(doc_event_type_score)
                
                #edag predict
                doc_decode_res = []
                for batch_idx, ins in enumerate(batch_test):
                    decode_res = []
                    for event_idx, event_type_pred in enumerate(doc_event_type_pred[batch_idx]):
                        if event_type_pred == 0 or doc_span_drange_list[batch_idx] is None:
                            decode_res.append(None)
                            continue

                        last_field_paths = [()]
                        for key, info in self.collate_info.items():
                            info['cur_mem_list'] = [info['doc_sent_emb'][batch_idx]]
                            info['prev_path2mem'] = {(): info['cur_mem_list']}
                        
                        for field_idx, field in enumerate(EVENT_TYPE_FIELDS_PAIRS[event_idx][1]):
                            cur_paths = []
                            for prev_path in last_field_paths:
                                # base pred
                                for key, info in self.collate_info.items():
                                    assert info['doc_span_drange_list'] == doc_span_drange_list
                                    self.model.load_state_dict(info['model_state_dict'])
                                    self.model.eval()
                                    span_emb = info['doc_span_emb'][batch_idx]
                                    prev_mem_list = info['prev_path2mem'][prev_path]
                                    prev_mem = torch.cat(prev_mem_list, dim=0)
                                    event_table = self.model.event_tables[event_idx]
                                    num_spans = span_emb.shape[0]

                                    field_query = event_table.field_queries[field_idx]
                                    span_cand_emb = span_emb + field_query
                                    cand_emb = torch.cat([span_cand_emb, prev_mem], dim=0).unsqueeze(0)
                                    cand_output = self.model.field_context_encoder(cand_emb, None).squeeze(0)
                                    span_output = cand_output[:num_spans]
                                    span_logp = event_table(batch_span_emb=span_output, field_idx=field_idx) # already log softmax
                                    span_score = np.exp(span_logp.detach().cpu().numpy())
                                    info['span_score'] = span_score
                                    info['span_output'] = span_output

                                #blending pred
                                blending_span_feature = []
                                for key, info in self.collate_info.items():
                                    blending_span_feature.append(info['span_score'])
                                blending_span_feature = np.hstack(blending_span_feature)
                                
                                if self.event2field2clf[event_idx][field_idx]['clf'] is not None:
                                    span_pred = self.event2field2clf[event_idx][field_idx]['clf'].predict(blending_span_feature)
                                else:
                                    neg = np.mean(blending_span_feature[:, ::2], axis=-1)
                                    pos = np.mean(blending_span_feature[:, 1::2], axis=-1)
                                    span_pred = (pos > neg).astype(np.int)



                                #update path
                                cur_span_idx_list = []
                                for cur_span_idx, pred_label in enumerate(span_pred):
                                    if pred_label == 1:
                                        cur_span_idx_list.append(cur_span_idx)
                                if len(cur_span_idx_list) == 0:
                                    cur_span_idx_list.append(None)

                                for key, info in self.collate_info.items():
                                    self.model.load_state_dict(info['model_state_dict'])
                                    self.model.eval()
                                    event_table = self.model.event_tables[event_idx]
                                    field_query = event_table.field_queries[field_idx]
                                    span_output = info['span_output']
                                    
                                    model_cur_paths = []
                                    for span_idx in cur_span_idx_list:
                                        cur_path = prev_path + (span_idx, )
                                        if span_idx is None:
                                            span_mem = field_query
                                        else:
                                            span_mem = span_output[span_idx].unsqueeze(0)
                                        cur_mem_list = list(prev_mem_list) #copy
                                        cur_mem_list.append(span_mem)
                                        info['prev_path2mem'][cur_path] = cur_mem_list
                                        model_cur_paths.append(cur_path)
                                    info['model_cur_paths'] = model_cur_paths
                                cur_paths.extend(model_cur_paths)
                            last_field_paths = cur_paths
                        
                        span_drange_list = doc_span_drange_list[batch_idx]
                        decode_paths = last_field_paths
                        event_res = []
                        for decode_path in decode_paths:
                            field_res = []
                            for span_idx in decode_path:
                                if span_idx is not None:
                                    field_res.append(span_drange_list[span_idx][1])
                                else:
                                    field_res.append(None)
                            event_res.append(field_res)
                        decode_res.append(event_res)
                    doc_decode_res.append(decode_res)
                total_decode_res.extend(doc_decode_res)
                pbar.update()
                # if batch_num > 10:
                #     break
                #break
            self.total_decode_res = total_decode_res
            return total_decode_res
    
    def save_decode_res(self, doc_decode_res=None):
        if doc_decode_res is None:
            doc_decode_res = self.total_decode_res

        self.dee_task.decode_drange2text(doc_decode_res, self.test)
        for ins, decode_res in zip(self.test, doc_decode_res):
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

        with open(self.config['test_result_file'], mode='w', encoding='utf-8') as f:
            for ins in self.test:
                write_obj = {}
                write_obj['doc_id'] = ins['doc_id']
                if 'events' not in ins:
                    break
                write_obj['events'] = ins['events']
                f.write(json.dumps(write_obj, ensure_ascii=False) + '\n')
        return self.test


            





if __name__ == '__main__':
    blend_config = {
        'ensemble_model_dir': 'output/save_model',
        'ensemble_model_eval_dir': 'output/save_eval',
        'skip_lower_epoch': 25,
        'ensemble_best_epochs': 3,
        'train_pkl': 'validate_doc.pkl',
        'test_pkl': 'test_doc.pkl',
        'test_result_file': 'result.txt',

        'random_seed': 666,
        'debug_data_num': None
    }
    dee_task = DocEETask(default_task_config)
    task = BlendingTask(blend_config, dee_task)
    task.collate_model()
    task.init_base_model()
    task.fit_ner(task.train)
    #task.ner_clf = pickle.load(open('blending_ner_model.pkl', mode='rb'))
    task.fit_edag(task.train)
    #task.event2field2clf = pickle.load(open('blending_edag_field_model.pkl', mode='rb'))
    #task.event_clf = pickle.load(open('blending_edag_event_model.pkl', mode='rb'))
    task.edag_predict()
    task.save_decode_res()
    