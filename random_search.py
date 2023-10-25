import argparse
import time
import pandas
from copy import deepcopy
from utils import *
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
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()


columns = ['name']
new_row = {}
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
            'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial']
models = model_detector_dict.keys()

if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
    print('Evaluated Datasets: ', datasets)

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset_name in datasets:
    for metric in ['AUROC', 'AUPRC', 'RecK', 'Time']:
        columns.append(dataset_name+'-'+metric)

results = pandas.DataFrame(columns=columns)
best_model_configs = {}
file_id = None

for model in models:
    model_result = {'name': model}
    best_model_configs[model] = {}
    for dataset_name in datasets:
        print('============Dataset {} Model {}=============='.format(dataset_name, model))
        auc_list, pre_list, rec_list = [], [], []
        set_seed()
        time_cost = 0
        best_val_score = 0
        train_config = {
            'device': args.device,
            'epochs': 100,
            'patience': 20,
            'metric': 'AUPRC',
            'inductive': False
        }
        data = Dataset(dataset_name)
        data.split(args.semi_supervised, 0)

        for t in range(args.trials):
            print("Dataset {}, Model {}, Trial {}, Time Cost {:.2f}".format(dataset_name, model, t, time_cost))
            if time_cost > 7200:  # 86400 Stop after 1 day
                break
            try:
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
                    best_troc, best_tprc, best_treck = test_score['AUROC'], test_score['AUPRC'], test_score['RecK']
                ed = time.time()
                time_cost += ed - st
                print("Current Val Best:{:.4f}; Test AUC:{:.4f}, PRC:{:.4f}, RECK:{:.4f}".format(
                    detector.best_score, test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            except:
                torch.cuda.empty_cache()
        print("best_model_config:", best_model_config)
        best_model_configs[model][dataset_name] = deepcopy(best_model_config)
        model_result[dataset_name+'-AUROC'] = best_troc
        model_result[dataset_name+'-AUPRC'] = best_tprc
        model_result[dataset_name+'-RecK'] = best_treck
        model_result[dataset_name+'-Time'] = time_cost/(t+1)
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    file_id = save_results(results, file_id)
