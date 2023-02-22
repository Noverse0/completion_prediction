from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from model import GCN
from utils import save_loss
from sklearn.metrics import classification_report, roc_auc_score

def train(args, train_dataloader, test_dataloader):
    device = torch.device("cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
        print('gpu is working')
    torch.cuda.empty_cache()
    if args.model == 'GCN':
        input_dim = list(
            train_dataloader.dataset[0][0].ndata['feature'][0].shape)[0]  # one_hot size
        output_dim = args.num_classes
        model = GCN(args.num_layers, input_dim,
                    args.hidden_dim, output_dim, device).to(device)

    ########### TODO other model define #############

    loss_func = nn.MSELoss()  # define loss function
    #loss_func = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    #optimizer = SGD(model.parameters(), lr=args.lr)
    
    train_loss = 0
    loss_record = []

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
            train_loss += loss.item()
            
            optimizer.step()
            
        train_loss = train_loss / len(train_dataloader.dataset)
        print('Epoch{}, train_loss {:.4f}'.format(
                epoch, train_loss))
        loss_record.append(train_loss)

    save_loss(loss_record, args.num_epochs, args.model, args.lr, args.num_layers)

    ######### eval ############
    labels = []
    pred_value = []
    
    model.eval()
    for graph, label in test_dataloader:
        graph = graph.to(device)
        label = label.type(torch.FloatTensor).to(device)
                    
        prediction = model(graph)
        pred_result = prediction.argmax()
        
        labels.append(label.argmax())
        pred_value.append(pred_result)
        
        #print(prediction, pred_result)
        
    print(classification_report(labels, pred_value))

    roc_score = roc_auc_score(labels, pred_value)
    
    print('auc {:.4f}'.format(
        roc_score))
            