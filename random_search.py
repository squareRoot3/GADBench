import argparse
import random
import time
import numpy as np
from detector import *
import pandas
import os
import json
from copy import deepcopy
from hyperparams import *
import warnings
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=100)
parser.add_argument('--semi_supervised', type=bool, default=False)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()

model_detector_dict = {
    'MLP': BaseGNNDetector,
    'KNN': KNNDetector,
    'SVM': SVMDetector,
    'RF': RFDetector,
    'XGBoost': XGBoostDetector,
    'XGBOD': XGBODDetector,

    'GCN': BaseGNNDetector,
    'SGC': BaseGNNDetector,
    'GIN': BaseGNNDetector,
    'GraphSAGE': BaseGNNDetector,
    'GAT': BaseGNNDetector,
    'GT': BaseGNNDetector,
    'ChebNet': BaseGNNDetector,

    'KNNGCN': KNNGCNDetector,
    'GAS': GASDetector,
    'GATSep': BaseGNNDetector,
    'BernNet': BaseGNNDetector,
    'AMNet': BaseGNNDetector,
    'BWGNN': BaseGNNDetector,
    'GHRN': GHRNDetector,
    'PCGNN': PCGNNDetector,
    'DCI': DCIDetector,
    # '':,
    'RFGraph': RFGraphDetector,
    'XGBGraph': XGBGraphDetector,
}

columns = ['name']
new_row = {}
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
            'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial']
models = param_space.keys()

if args.datasets is not None:
    st, ed = args.datasets.split('-')
    datasets = datasets[int(st):int(ed)+1]
    print('Evaluated Datasets: ', datasets)

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset_name in datasets:
    for metric in ['AUROC', 'AUPRC', 'RecK', 'Time']:
        columns.append(dataset_name+'-'+metric)

results = pandas.DataFrame(columns=columns)
best_model_configs = {}
file_knt = None

for model in models:
    model_result = {'name': model}
    best_model_configs[model] = {}
    for dataset_name in datasets:
        print('============Dataset {} Model {}=============='.format(dataset_name, model))
        # if model == 'XGBOD' and dataset_name in ['elliptic', 'dgraphfin']:
        #     continue
        auc_list, pre_list, rec_list = [], [], []
        set_seed()
        time_cost = 0
        best_val_score = 0
        train_config = {
            'device': args.device,
            'epochs': 100,
            'patience': 20,
            'metric': 'AUPRC',
        }
        data = Dataset(dataset_name)
        data.split(args.semi_supervised, 0)

        for t in range(args.trials):
            print("Dataset {}, Model {}, Trial {}, Time Cost {:.2f}".format(dataset_name, model, t, time_cost))
            if time_cost > 3600:  # Stop after 2 hours
                break
            # try:
            model_config = sample_param(model, dataset_name, t)
            detector = model_detector_dict[model](train_config, model_config, data)
            train_config['seed'] = seed_list[t]
            st = time.time()
            print("model_config: ", model_config)
            test_score = detector.train()
            if detector.best_score > best_val_score:
                print("****current best score****")
                best_val_score = detector.best_score
                best_model_config = deepcopy(model_config)
                best_tauc, best_tpre, best_treck = test_score['AUROC'], test_score['AUPRC'], test_score['RECK']
            ed = time.time()
            time_cost += ed - st
            print("Current Val Best:{:.4f}; Test AUC:{:.4f}, PRC:{:.4f}, RECK:{:.4f}".format(
                detector.best_score, test_score['AUROC'], test_score['AUPRC'], test_score['RECK']))
            # except:
            #     torch.cuda.empty_cache()

        print("best_model_config:", best_model_config)
        best_model_configs[model][dataset_name] = deepcopy(best_model_config)
        model_result[dataset_name+'-AUROC'] = best_tauc
        model_result[dataset_name+'-AUPRC'] = best_tpre
        model_result[dataset_name+'-RecK'] = best_treck
        model_result[dataset_name+'-Time'] = time_cost/t
    if file_knt is None:
        file_knt = 0
        while os.path.exists('results/{}.xlsx'.format(file_knt)):
            file_knt += 1
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    print(results)
    print(best_model_configs)
    results.transpose().to_excel('results/{}.xlsx'.format(file_knt))

with open("results/best_model_configs_{}.json".format(file_knt), 'w') as f:
    json.dump(best_model_configs, f)
results.transpose().to_excel('results/{}.xlsx'.format(file_knt))
print('save to file ID: {}'.format(file_knt))