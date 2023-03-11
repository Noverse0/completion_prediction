import argparse
import torch as th
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from graph_builder import OuladDataset, Oulad_main_Dataset
from utils import graph_print
from trainer import graph_linkprediction_train


def main(args):
    # Naver
    if args.dataset == 'naver':
        print('naver')
        # dataset = NaverDataset(arg.days)
    # OULAD
    elif args.dataset == 'oulad':
        dataset = Oulad_main_Dataset(args.days, args.split_ratio)
        if args.print:
            graph_print(g)
            return
    else:
        raise ValueError()
        
    dataloader = GraphDataLoader(dataset)
    graph_linkprediction_train(args, dataloader)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="default is 30 days")
    parser.add_argument("--num_classes", type=int, default=2, help="number of class(P/F)")
    parser.add_argument("--hidden_dim", type=int, default=16, help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use(naver or oulad)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size default 32")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="default train ratio is 0.8")
    parser.add_argument("--model", type=str, default='GCN', help="default Model is GCN")
    parser.add_argument("--print", type=bool, default=False, help="graph print option")
    parser.add_argument("--num_layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--threshold", type=float, default=0.5, help="classification threshold")

    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        default=50,
        help="number of training epochs",
    )
    
    parser.add_argument(
        "--model_path", type=str, default=None, help="path for save the model"
    )
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")
    parser.set_defaults(validation=True)
    
    args = parser.parse_args()
    print(args)
    main(args)
    
