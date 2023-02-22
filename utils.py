import networkx as nx
import dgl
import matplotlib.pyplot as plt
import numpy as np

def graph_print(dataset):
    index = 0
    for g, label in dataset:
        nx_graph = dgl.to_networkx(g)

        # color and pos
        color = []
        pos = {}
        activity_x = 2
        activity_y = 0
        date_x = 0
        date_y = 0

        for i in range(len(g.ndata['node_type'])):
            node = list(nx_graph.nodes())[i]
            if list(g.ndata['node_type'][i]) == [1,0,0]:
                color.append(1)
                pos[node] = (activity_x , activity_y)
                activity_y += 1
            elif list(g.ndata['node_type'][i]) == [0,1,0]:
                color.append(2)
                pos[node] = (activity_x , activity_y)
                activity_y += 1
            else:
                color.append(3)
                pos[node] = (date_x , date_y)
                date_y += 2

        if label == [1]:
            result = 'PASS'
        else:
            result = 'FAIL'
        
        title = '/graph_' + str(index)
        plt.figure(figsize=(5,10))
        nx.draw(nx_graph, pos, node_color=color)
        plt.title('Yello is Date Node, Purple is Activity Node, Green is Assessment Node  This student is' + result)
        plt.show()
        plt.savefig('./images/oulad' + title + '.png')
        index += 1
        
def save_loss(loss_record, num_epochs, model, lr, num_layers):
    plt.plot(np.arange(num_epochs), loss_record)
    plt.savefig('./images/' + model + '_loss' + '_epoch_' + num_epochs + '_layers_' + num_layers + '_lr_' + lr + '.png')