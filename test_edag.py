from utils import measure_event_table_filling
import json
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import matplotlib.pyplot as plt
import torch

eval_target = []
for output_dir in os.listdir('output_edag'):
    if output_dir.startswith('save_eval'):
        eval_target.append((os.path.join('output_edag', output_dir), 'test', 'black'))
eval_target.sort(key=lambda x: x[0])
max_epoch = 60
min_epoch = 30
res = {}
handles = []
for eval_dir_path, eval_label, color in eval_target:
    save_eval_dir = os.listdir(eval_dir_path)
    _ner_res = []
    _ee_res = []
    for eval_file in save_eval_dir:
        if eval_file.startswith('eval-ner') and eval_file.endswith('json'):
            epoch = int(eval_file.split('-')[-1].split('.')[0])
            if epoch >= max_epoch or epoch < min_epoch:
                continue
            eval_json = json.load(open(os.path.join(eval_dir_path, eval_file), mode='r', encoding='utf-8'))
            f1 = eval_json['micro_f1']
            _ner_res.append((epoch, f1))
        elif eval_file.startswith('eval') and eval_file.endswith('json'):
            epoch = int(eval_file.split('-')[-1].split('.')[0])
            if epoch >= max_epoch or epoch < min_epoch:
                continue
            eval_json = json.load(open(os.path.join(eval_dir_path, eval_file), mode='r', encoding='utf-8'))
            f1 = eval_json[-1]['MicroF1']
            _ee_res.append((epoch, f1))
    
    store_dict = { 'setting': None }

    model_dir_path = eval_dir_path.replace('eval', 'model')
    if os.path.exists(model_dir_path):
        model_dir = os.listdir(model_dir_path)
        model_path = model_dir[0]
        store_dict = torch.load(os.path.join(model_dir_path, model_path), map_location='cpu')

    _ner_res = sorted(_ner_res, key=lambda x: x[0], reverse=True)
    _ee_res = sorted(_ee_res, key=lambda x: x[0], reverse=True)
    epochs = list(map(lambda x: x[0], _ee_res))
    micro_f1 = list(map(lambda x: x[1], _ee_res))
    res[eval_dir_path] = {
        'ner_res': _ner_res, 'ee_res': _ee_res, 'epochs': epochs, 'micro_f1': micro_f1, 'max_micro_f1': max(micro_f1),
        'mean_micro_f1': np.mean(micro_f1),
        'setting': store_dict['setting']
    }

    handle = plt.plot(epochs, micro_f1, color=color)[0]
    handles.append(handle)
f1_res = list(map(lambda x: (x[0], x[1]['max_micro_f1']), res.items()))
f1_res.sort(key=lambda x: x[1])
for model_name, score in f1_res:
    print(model_name, score)

############################## eval ner
ner_eval_target = [
    ('output_edag/save_eval_transformer_parser', 'edag', 'red'),
    ('output_edag/save_eval_transformer_cw', 'edag', 'red'),
    ('output_edag/save_eval_transformer', 'graph', 'blue'),
    ('output_edag/save_eval_transformer_pos', 'test', 'black'),
    ('output_edag/save_eval_transformer_total', 'test', 'black'),
    ('output_edag/save_eval_bert&lstm', 'test', 'black'),
    #('output_edag/save_eval_transformer_total', 'test', 'black'),
]

ner_eval_target = []
for output_dir in os.listdir('output_edag_ner'):
    if output_dir.startswith('save_eval'):
        ner_eval_target.append((os.path.join('output_edag', output_dir), 'test', 'black'))
ner_eval_target.sort(key=lambda x: x[0])
max_epoch = 20
ner_res = {}
handles = []
for eval_dir_path, eval_label, color in ner_eval_target:
    save_eval_dir = os.listdir(eval_dir_path)
    res = []
    micro_f1 = []
    epochs = []
    for eval_file in save_eval_dir:
        if eval_file.endswith('json'):
            epoch = int(eval_file.split('-')[-1].split('.')[0])
            if epoch >= max_epoch:
                continue
            eval_json = json.load(open(os.path.join(eval_dir_path, eval_file), mode='r', encoding='utf-8'))
            f1 = eval_json['micro_f1']
            res.append((epoch, f1))
    
    store_dict = { 'setting': None }

    model_dir_path = eval_dir_path.replace('eval', 'model')
    if os.path.exists(model_dir_path):
        model_dir = os.listdir(model_dir_path)
        model_path = model_dir[0]
        store_dict = torch.load(os.path.join(model_dir_path, model_path), map_location='cpu')

    res = sorted(res, key=lambda x: x[0], reverse=True)
    epochs = list(map(lambda x: x[0], res))
    micro_f1 = list(map(lambda x: x[1], res))
    ner_res[eval_dir_path] = {
        'res': res, 'epochs': epochs, 'micro_f1': micro_f1, 'max_micro_f1': max(micro_f1),
        'setting': store_dict['setting']
    }

    handle = plt.plot(epochs, micro_f1, color=color)[0]
    handles.append(handle)
ner_f1_res = list(map(lambda x: (x[0], x[1]['max_micro_f1']), ner_res.items()))
ner_f1_res.sort(key=lambda x: x[1])
for model_name, score in ner_f1_res:
    print(model_name, score)


# plt.legend(handles, list(map(lambda x: x[1], eval_target)))
# plt.xlabel('epoch')
# plt.ylabel('mirco-f1')
# plt.savefig('test.png')