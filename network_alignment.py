from utils.dataset import Dataset
from algorithms import *
from evaluation.metrics import get_statistics
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="graph_data/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="graph_data/douban/offline/graphsage/")
    parser.add_argument('--groundtruth', default="graph_data/douban/dictionaries/groundtruth")
    subparsers = parser.add_subparsers(dest="algorithm")
    # CINA
    parser_CINA = subparsers.add_parser("CINA", help="CINA algorithm")
    parser_CINA.add_argument('--cuda', action="store_false", help="store_false if cpu")
    parser_CINA.add_argument('--embedding_dim', default=192, type=int)
    parser_CINA.add_argument('--CINA_epochs', default=20, type=int)
    parser_CINA.add_argument('--lr', default=0.02, type=float)
    parser_CINA.add_argument('--train_dict', type=str)
    parser_CINA.add_argument('--hCINA_lr', default=0.02, type=float)
    # refinement
    parser_CINA.add_argument('--refinement_epochs', default=10, type=int)
    parser_CINA.add_argument('--refine', action="store_true", help="wheather to use refinement step")
    parser_CINA.add_argument('--threshold_refine', type=float, default=0.95,
                             help="The threshold value to get stable candidates")
    parser_CINA.add_argument('--threshold', default=0.01, type=float,help="Threshold of for sharpenning")
    # PALE
    parser_CINA.add_argument('--pale_lr', type=float, default=0.01)
    parser_CINA.add_argument('--pale_epochs', type=int, default=500)
    parser_CINA.add_argument('--pale_emb_batchsize', type=int, default=512)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time.time()
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    algorithm = args.algorithm

    model = CINA(source_dataset, target_dataset, args)

    S = model.align()

    end_time = time.time()
    run_time = end_time - start_time
    print("run_time:",run_time)

    acc, MAP, Hit, AUC, top5, top10, top20, top30 = get_statistics(S, groundtruth, get_all_metric=True)
    print("Top_1: {:.4f}".format(acc))
    print("Top_5: {:.4f}".format(top5))
    print("Top_10: {:.4f}".format(top10))
    print("Top_20: {:.4f}".format(top20))
    print("Top_30: {:.4f}".format(top30))
    print("Hit: {:.4f}".format(Hit))
    print("MAP: {:.4f}".format(MAP))
    print("AUC: {:.4f}".format(AUC))
    print("-" * 100)

    result = acc, top5, top10, top20, top30
