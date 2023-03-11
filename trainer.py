from tqdm import tqdm
import torch
import dgl
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
            
            
def graph_linkprediction_train(args, train_dataloader):
    device = torch.device("cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
        print('gpu is working')
    torch.cuda.empty_cache()
    
    g = train_dataloader.dataset[0]
    train_pos_g = train_dataloader.dataset[1]
    train_neg_g = train_dataloader.dataset[2]
    test_pos_g = train_dataloader.dataset[3]
    test_neg_g = train_dataloader.dataset[4]
        
    if args.model == 'GCN':
        input_dim = g.ndata['feature'].shape[1]
        #output_dim = args.num_classes
        model = GCN(args.num_layers, input_dim,
                    args.hidden_dim, device).to(device)

    ########### TODO other model define #############

    pred = MLPPredictor(args.hidden_dim).to(device)

    # ----------- 3. set up loss and optimizer -------------- #
    # in this case, loss will in training loop
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # ----------- training -------------------------------- #
    for e in range(args.num_epochs):
        # forward
        h = model(dgl.add_self_loop(g).to(device))
        
        pos_score = pred(train_pos_g.to(device), h)
        neg_score = pred(train_neg_g.to(device), h)
        loss = compute_loss(pos_score, neg_score)
        #print(loss)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))
            
    # -----------  check results ------------------------ #
    with torch.no_grad():
        pos_score = pred(test_pos_g.to(device), h)
        neg_score = pred(test_neg_g.to(device), h)
        
        pass_accuracy, fail_accuracy = 0, 0
        
        predicted_pass = []
        for i in pos_score.tolist():
            if i > args.threshold:
                predicted_pass.append(1)
                pass_accuracy += 1
            else:
                predicted_pass.append(0)
                
        predicted_fail = []
        for i in neg_score.tolist():
            if i > args.threshold:
                predicted_fail.append(1)
            else:
                predicted_fail.append(0)
                fail_accuracy += 1
        
        print('AUC using probability', compute_auc(pos_score, neg_score))
        print('AUC using classification', compute_auc(torch.tensor(predicted_pass), torch.tensor(predicted_fail)))
        print('accuracy', (pass_accuracy+fail_accuracy)/(len(pos_score)+len(neg_score)))
        print('pass accuracy', pass_accuracy/len(pos_score))
        print('fail accuracy', fail_accuracy/len(neg_score))