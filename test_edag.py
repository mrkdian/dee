from utils import measure_event_table_filling
import json
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import matplotlib.pyplot as plt
import torch

ner_eval_target = [
    ('output_edag/save_eval_rbt3_parser', 'edag', 'red'),
    ('output_edag/save_eval_rbt3_pos', 'edag', 'red'),
    ('output_edag/save_eval_rbt3_cw', 'edag', 'red'),
    ('output_edag/save_eval_rbt3', 'graph', 'blue'),
    #('output_edag/save_eval', 'test', 'black')
]
ner_res = {}
handles = []
for eval_dir_path, eval_label, color in ner_eval_target:
    model_dir_path = eval_dir_path.replace('eval', 'model')
    model_dir = os.listdir(model_dir_path)
    save_eval_dir = os.listdir(eval_dir_path)
    res = []
    micro_f1 = []
    epochs = []
    for eval_file in save_eval_dir:
        if eval_file.endswith('json'):
            epoch = int(eval_file.split('-')[-1].split('.')[0])
            eval_json = json.load(open(os.path.join(eval_dir_path, eval_file), mode='r', encoding='utf-8'))
            f1 = eval_json['micro_f1']
            res.append((epoch, f1))
    
    
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
print(1)
# plt.legend(handles, list(map(lambda x: x[1], eval_target)))
# plt.xlabel('epoch')
# plt.ylabel('mirco-f1')
# plt.savefig('test.png')