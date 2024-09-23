import sys 
sys.path.append('../')

import naslib


from naslib.predictors import ZeroCost
from naslib.search_spaces import NasBench201SearchSpace, NATSBenchSizeSearchSpace, NasBench101SearchSpace, NasBench301SearchSpace
from naslib.utils import get_train_val_loaders, get_project_root
from fvcore.common.config import CfgNode
from tqdm import tqdm

# Create configs required for get_train_val_loaders
config = {
    'dataset': 'cifar10', # Dataset to loader: can be cifar100, svhn, ImageNet16-120, jigsaw, class_object, class_scene, or autoencoder (the last four are TNB101 datasets)
    'data': str(get_project_root()) + '/data', # path to naslib/data
    'search': {
        'seed': 9001, # Seed to use in the train, validation and test dataloaders
        'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
        'batch_size': 100, # batch size of the dataloaders
    }
}
config = CfgNode(config)

# Get the dataloaders
train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)


from naslib.search_spaces.core import Metric
from naslib.utils import compute_scores, get_dataset_api

# Sample 50 random architectures, query their performances
n_graphs = 15000
models = []
val_accs = []
zc_scores_nasi = []
zc_scores_monet = []
zc_scores_synflow = []
zc_scores_nwot = []

print('Loading NAS-Bench-201 API...')
api = get_dataset_api(search_space='nasbench201', dataset='cifar10')

print(f'Sampling {n_graphs} NAS-Bench-201 models...')
for _ in tqdm(range(n_graphs)):
    graph = NasBench201SearchSpace()
    graph.sample_random_architecture()
    graph.parse()

    models.append(graph)

zc_predictor_nasi = ZeroCost(method_type="ntk_trace")
zc_predictor_monet = ZeroCost(method_type="monet")
zc_predictor_synflow = ZeroCost(method_type="synflow")
zc_predictor_nwot = ZeroCost(method_type="nwot")

print('Querying validation performance and scores for all models')
for graph in tqdm(models):
    acc = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=api)
    val_accs.append(acc)

    score_nasi = zc_predictor_nasi.query(graph=graph, dataloader=train_loader)
    zc_scores_nasi.append(score_nasi)

    score_monet = zc_predictor_monet.query(graph=graph, dataloader=train_loader)
    zc_scores_monet.append(score_monet)
 
    score_synflow = zc_predictor_synflow.query(graph=graph, dataloader=train_loader)
    zc_scores_synflow.append(score_synflow) 

    score_nwot = zc_predictor_nwot.query(graph=graph, dataloader=train_loader)
    zc_scores_nwot.append(score_nwot) 

# We now compute the correlation between val_accs (ground truth) and zc_scores (proxy scores)
correlations_nasi = compute_scores(ytest=val_accs, test_pred=zc_scores_nasi)
correlations_monet = compute_scores(ytest=val_accs, test_pred=zc_scores_monet)
correlations_synflow = compute_scores(ytest=val_accs, test_pred=zc_scores_synflow)
correlations_nwot = compute_scores(ytest=val_accs, test_pred=zc_scores_nwot)

# Extract the results
# kendalltau_corr = correlations['kendalltau']
# spearman_corr = correlations['spearman']
# pearson_corr = correlations['pearson']

corr_nasi = correlations_nasi["spearman"]
corr_monet = correlations_monet["spearman"]
corr_synflow = correlations_synflow["spearman"]
corr_nwot = correlations_nwot["spearman"]

print('Correlations between validation accuracies (ground truth) and Zero Cost predictor scores (prediction): ')
print('Spearman correlation NASI :', corr_nasi)
print('Spearman correlation MONET :', corr_monet)
print('Spearman correlation SYNFLOW :', corr_synflow)
print('Spearman correlation NASWOT :', corr_nwot)
