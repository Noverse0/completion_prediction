from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from model import GCN, MLPPredictor
from utils import save_loss
from sklearn.metrics import classification_report, roc_auc_score
import torch.nn.functional as F

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def graph_classification_train(args, train_dataloader, test_dataloader):
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
            
            
            
def graph_linkprediction_train(args, train_dataloader):
    device = torch.device("cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
        print('gpu is working')
    torch.cuda.empty_cache()
    if args.model == 'GCN':
        input_dim = g.ndata['feature'].shape[1]
        output_dim = args.num_classes
        model = GCN(args.num_layers, input_dim,
                    args.hidden_dim, output_dim, device).to(device)

    ########### TODO other model define #############

    #model = GraphSAGE(g.ndata['feature'].shape[1], 16)

    model = GCN(2, g.ndata['feature'].shape[1], 16, 16)

    # You can replace DotPredictor with MLPPredictor.
    pred = MLPPredictor(16)
    #pred = DotPredictor()

    # ----------- 3. set up loss and optimizer -------------- #
    # in this case, loss will in training loop
    optimizer = Adam(model.parameters(), lr=args.lr)

    # ----------- 4. training -------------------------------- #
    for e in range(500):
        # forward
        h = model(dgl.add_self_loop(g))
        
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        #print(loss)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))
            
    # ----------- 5. check results ------------------------ #
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        
        pass_accuracy, fail_accuracy = 0, 0
        
        predicted_pass = []
        for i in pos_score.tolist():
            if i > 0.5:
                predicted_pass.append(1)
                pass_accuracy += 1
            else:
                predicted_pass.append(0)
                
        predicted_fail = []
        for i in neg_score.tolist():
            if i > 0.5:
                predicted_fail.append(1)
            else:
                predicted_fail.append(0)
                fail_accuracy += 1
        
        print('AUC', compute_auc(pos_score, neg_score))
        print('AUC', compute_auc(torch.tensor(predicted_pass), torch.tensor(predicted_fail)))
        print('accuracy', (pass_accuracy+fail_accuracy)/(len(pos_score)+len(neg_score)))
        print('pass accuracy', pass_accuracy/len(pos_score))
        print('fail accuracy', fail_accuracy/len(neg_score))