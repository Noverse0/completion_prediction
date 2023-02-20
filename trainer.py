from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from model import GCN




def train(args, train_dataloader, test_dataloader):
    device = torch.device("cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    if args.model == 'GCN':
        input_dim = list(
            train_dataloader.dataset[0][0].ndata['feature'][0].shape)[0]  # one_hot size
        output_dim = args.num_classes
        model = GCN(args.num_layers, input_dim,
                    args.hidden_dim, output_dim, device).to(device)
        
    ########### TODO other model define #############

    loss_func = nn.CrossEntropyLoss()  # define loss function
    optimizer = Adam(model.parameters(), lr=args.lr)

    print("start")
    for epoch in range(args.num_epochs):
        model.train()
        for graph, label in tqdm(train_dataloader):
            graph = graph.to(device)
            label = label.type(torch.FloatTensor).to(device)
            
            prediction = model(graph)

            optimizer.zero_grad()
            loss = loss_func(prediction, label)
            loss.backward()

            optimizer.step()
        print('Epoch{}, train_loss {:.4f}'.format(
                epoch, loss))

    ######### eval ############
    tn, tp, fn, fp = 0, 0, 0, 0
    model.eval()
    for graph, label in test_dataloader:
        graph = graph.to(device)
        label = label.type(torch.FloatTensor).to(device)
                    
        prediction = model(graph)
        pred_result = prediction.argmax(1)
        
        if int(label[0][0]) == 1: # pass 일때
            if pred_result == label[0][0]: # 맞춤
                tp += 1
            else: 
                fn += 1
        else:
            if pred_result == label[0][0]:
                tn += 1
            else:
                fp += 1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2/((1/precision)+(1/recall))
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    
    print('accuracy {:.4f}, f1 score {:.4f}, precision {:.4f}, recall {:.4f}'.format(
        accuracy, f1, precision, recall))
            
            