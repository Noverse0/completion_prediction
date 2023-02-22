import argparse
import torch as th
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from graph_builder import OuladDataset
from utils import graph_print
from trainer import train


def main(args):
    # Naver
    if args.dataset == 'naver':
        print('naver')
        # dataset = NaverDataset(arg.days)
    # OULAD
    elif args.dataset == 'oulad':
        dataset = OuladDataset(args.days)
        if args.print:
            graph_print(dataset)
            return
    else:
        raise ValueError()
        
    # cuda test
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.manual_seed_all(0)
        th.cuda.set_device(args.gpu)
        
    num_examples = len(dataset)
    num_train = int(num_examples * args.split_ratio)

    train_sampler = SubsetRandomSampler(th.arange(num_train))
    test_sampler = SubsetRandomSampler(th.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, drop_last=False)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, drop_last=False)
    
    train(args, train_dataloader, test_dataloader)
    
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
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of propagation rounds"
    )
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
    
