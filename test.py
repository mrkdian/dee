from utils import measure_event_table_filling
import json
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import matplotlib.pyplot as plt
from pyltp import Segmentor



# test = []
# for line in open('output/save_test_edag_doc_enc_graph/test-21.txt-with_content', mode='r', encoding='utf-8'):
#     test.append(json.loads(line))

# print(test)

eval_target = [
    # ('output/save_eval_edag_doc_enc', 'edag', 'red'),
    #('output/save_eval_xlnet_edag', 'graph', 'blue'),
    ('output/save_eval', 'test', 'black')
]
handles = []
for eval_dir_path, eval_label, color in eval_target:
    save_eval_dir = os.listdir(eval_dir_path)
    res = []
    micro_f1 = []
    epochs = []
    for eval_file in save_eval_dir:
        if eval_file.endswith('json'):
            epoch = int(eval_file.split('-')[1].split('.')[0])
            eval_json = json.load(open(os.path.join(eval_dir_path, eval_file), mode='r', encoding='utf-8'))
            f1 = eval_json[-1]['MicroF1']
            res.append((epoch, f1))
    res = sorted(res, key=lambda x: x[0], reverse=True)
    epochs = list(map(lambda x: x[0], res))
    micro_f1 = list(map(lambda x: x[1], res))

    handle = plt.plot(epochs, micro_f1, color=color)[0]
    handles.append(handle)
plt.legend(handles, list(map(lambda x: x[1], eval_target)))
plt.xlabel('epoch')
plt.ylabel('mirco-f1')
plt.savefig('test.png')


eval_dir_path = 'output/save_eval'
save_eval_dir = os.listdir(eval_dir_path)

obj_res = []
for eval_file in save_eval_dir:
    if eval_file.endswith('pkl'):
        epoch = int(eval_file.split('-')[-1].split('.')[0])
        eval_obj = pickle.load(open(os.path.join(eval_dir_path, eval_file), mode='rb'))
        best_bound = 0
        best_f1 = -1
        bounds = np.arange(0, 1, 0.1)
        for bound in bounds:
            gt = []
            pred = []
            for _gt, _pred in zip(eval_obj['span2span_gt'], eval_obj['span2span_pred']):
                _pred = np.array(_pred)
                _gt = _gt.reshape(-1)
                _pred = _pred.reshape(-1)
                assert _gt.shape[0] == _pred.shape[0]
                gt.append(_gt)
                pred.append(_pred)
            if len(gt) < 1:
                continue
            gt = np.hstack(gt)
            pred = np.hstack(pred) > bound
            p = precision_score(gt, pred)
            r = recall_score(gt, pred)
            f1 = f1_score(gt, pred)
            acc = accuracy_score(gt, pred)
            if f1 > best_f1:
                best_bound = bound
            best_f1 = max(best_f1, f1)
        obj_res.append((epoch, best_f1, best_bound, acc, p, r))
obj_res.sort(key=lambda x: x[1], reverse=True)
