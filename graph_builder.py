import pandas as pd
import numpy as np
import dgl
from dgl.data import DGLDataset
import torch
from tqdm import tqdm
import os
from dgl import save_graphs, load_graphs

def oulad_load():
    vle = pd.read_csv('data/archive/vle.csv')
    studentVle = pd.read_csv('data/archive/studentVle.csv')
    studentRegistration = pd.read_csv('data/archive/studentRegistration.csv')
    studentAssessment = pd.read_csv('data/archive/studentAssessment.csv')
    studentInfo = pd.read_csv('data/archive/studentInfo.csv')
    assessments = pd.read_csv('data/archive/assessments.csv')
    
    return vle, studentVle, studentRegistration, studentAssessment, studentInfo, assessments

def one_hot_encoding(n, n_to_index):
    one_hot_vector = [0]*(len(n_to_index))
    index = n_to_index[n]
    one_hot_vector[index] = 1
    return one_hot_vector

class OuladDataset(DGLDataset):
    def __init__(self, days):
        self.days = days
        super().__init__(name='oulad')
        
    def process(self):
        graph_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        return
        
        vle, studentVle, studentRegistration, studentAssessment, studentInfo, assessments = oulad_load()

        # pass and distinction = Completion, fail and withdrawn = Dropout
        studentInfo['completion_status'] = list(map(lambda x: 'Completion' if (x == 'Pass') or (x == 'Distinction') else 'Dropout', studentInfo['final_result']))
        studentInfo['graph_name'] = studentInfo['code_module'] + '_' + studentInfo['code_presentation'] + '_' + studentInfo['id_student'].astype(str)
    
        # make graph name module_presentation_userid
        studentAssessment = pd.merge(studentAssessment, assessments, how='left', on='id_assessment')
        studentAssessment['graph_name'] = studentAssessment['code_module'] + '_' + studentAssessment['code_presentation'] + '_' + studentAssessment['id_student'].astype(str)
        studentVle['graph_name'] = studentVle['code_module'] + '_' + studentVle['code_presentation'] + '_' + studentVle['id_student'].astype(str)
        studentAssessment['assessment_date'] = studentAssessment['id_assessment'].astype(str) + '_' + studentAssessment['date'].astype(str)
   
        # calculate assessment click_sum
        assessment_group = studentAssessment.groupby(['graph_name', 'assessment_date']).nunique()
        assessment_group.rename(columns = {'id_assessment' : 'sum_click'}, inplace = True)
        studentAssessment = pd.merge(studentAssessment, assessment_group['sum_click'], how='left', on=['graph_name','assessment_date'])

        # node type activity = 0, assessment = 1
        studentVle['activity'], studentVle['assessment'] = 0, 1
        studentAssessment['activity'], studentAssessment['assessment'] = 1, 0
        studentVle.rename(columns = {'id_site' : 'id_activity'}, inplace = True)
        studentAssessment.rename(columns = {'id_assessment' : 'id_activity'}, inplace = True)

        # make one-hot vector using activity_type, assessment_type
        activity_onehot_list = list(vle['activity_type'].unique())
        assessment_onehot_list = list(studentAssessment['assessment_type'].unique())
        
        activity_zero_list = [0 for i in range(len(activity_onehot_list))]
        assessment_zero_list = [0 for i in range(len(assessment_onehot_list))]
        
        activity_to_index = {n : index for index, n in enumerate(activity_onehot_list)}
        assessment_to_index = {n : index for index, n in enumerate(assessment_onehot_list)}
        
        # activity one-hot
        activity_node_feature, assessment_node_feature = [], []
        studentVle = pd.merge(studentVle, vle[['id_site', 'activity_type']], left_on='id_activity', right_on='id_site')
        for index, activity_type in enumerate(studentVle['activity_type']):
            activity_node_feature.append(one_hot_encoding(activity_type, activity_to_index))
            assessment_node_feature.append(assessment_zero_list)
        studentVle['activity_node_feature'] = activity_node_feature
        studentVle['assessment_node_feature'] = assessment_node_feature
        
        # assessment one-hot
        activity_node_feature, assessment_node_feature = [], []
        for index, assessment_type in enumerate(studentAssessment['assessment_type']):
            activity_node_feature.append(activity_zero_list)
            assessment_node_feature.append(one_hot_encoding(assessment_type, assessment_to_index))
        studentAssessment['activity_node_feature'] = activity_node_feature
        studentAssessment['assessment_node_feature'] = assessment_node_feature
        
        activity = pd.concat([studentVle[['graph_name', 'date', 'sum_click', 'activity', 'assessment', 'id_activity', 'activity_node_feature', 'assessment_node_feature']], 
                              studentAssessment[['graph_name', 'date', 'sum_click', 'activity', 'assessment', 'id_activity', 'activity_node_feature', 'assessment_node_feature']]])

        # Leave only n days after first interaction
        graph_group = activity.groupby('graph_name').agg(['min'])
        activity = pd.merge(activity, graph_group['date'], how='left', on='graph_name')
        activity['new_date'] = activity['date'] - activity['min']
        activity = activity[activity['new_date'] < self.days]
        
        # make date one-hot
        date_onehot_list = list(range(0, self.days, 1))
        date_to_index = {n : index for index, n in enumerate(date_onehot_list)}
        date_node_feature = []
        for index, date_type in enumerate(activity['new_date']):
            date_node_feature.append(one_hot_encoding(date_type, date_to_index))
        activity['date_node_feature'] = date_node_feature
        
        
        ##---------------make a graph-----------------------##
        self.graphs = []
        self.labels = []
        
        # Create a graph for each graph ID from the edges table
        label_dict = {}
        for _, row in studentInfo.iterrows():
            label_dict[row['graph_name']] = row['completion_status']

        # For the edges, first group the table by graph IDs.
        edges_group = activity.groupby('graph_name')

        # For each graph ID...
        for graph_name in tqdm(edges_group.groups):
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_name)
            
            node_set = {}
            index = 0
            for i in list(set(edges_of_id['id_activity'])) + list(set(edges_of_id['new_date'])):
                node_set[i] = index
                index += 1

            node_id = {node : index for index, node in enumerate(node_set)}

            src_node, dst_node, edge_feature = [], [], []
            for i in range(len(edges_of_id)):
                src_node.append(node_id[edges_of_id['id_activity'].to_numpy()[i]])
                dst_node.append(node_id[edges_of_id['new_date'].to_numpy()[i]])

            for i in range(len(edges_of_id)):
                dst_node.append(node_id[edges_of_id['id_activity'].to_numpy()[i]])
                src_node.append(node_id[edges_of_id['new_date'].to_numpy()[i]])
                
            g = dgl.graph((src_node, dst_node))

            # add node feature
            feature, node_type, date_node_feature, activity_node_feature, assessment_node_feature = [], [], [], [], []
            for node in list(node_id.keys()):
                if node < 30: # date node
                    date_node_feature.append(edges_of_id['date_node_feature'].to_numpy()[[np.where(edges_of_id['new_date'].to_numpy() == node)[0][0]]][0])
                    activity_node_feature.append(activity_zero_list)
                    assessment_node_feature.append(assessment_zero_list)
                    node_type.append([0,0,1])
                else:
                    date_node_feature.append([0 for i in range(30)])
                    activity_node_feature.append(edges_of_id['activity_node_feature'].to_numpy()[[np.where(edges_of_id['id_activity'].to_numpy() == node)[0][0]]][0])
                    assessment_node_feature.append(edges_of_id['assessment_node_feature'].to_numpy()[[np.where(edges_of_id['id_activity'].to_numpy() == node)[0][0]]][0])
                    if edges_of_id['assessment_node_feature'].to_numpy()[[np.where(edges_of_id['id_activity'].to_numpy() == node)[0][0]]][0] == [0,0,0]:
                        node_type.append([1,0,0]) # it is activity feature
                    else:
                        node_type.append([0,1,0])
                feature.append(node_type[-1] + activity_node_feature[-1] + assessment_node_feature[-1] + date_node_feature[-1])

            g.ndata['feature'] = torch.tensor(feature)
            g.ndata['node_type'] = torch.tensor(node_type)
            g.ndata['date_node_feature'] = torch.tensor(date_node_feature)
            g.ndata['activity_node_feature'] = torch.tensor(activity_node_feature)
            g.ndata['assessment_node_feature'] = torch.tensor(assessment_node_feature)

            g.edata['edge_feature'] = torch.FloatTensor(list(edges_of_id['sum_click']) + list(edges_of_id['sum_click']))
            
            # edge date with date
            date_src = list(set(edges_of_id['new_date']))[:-1]
            date_dst = list(set(edges_of_id['new_date']))[1:]
            for i in range(len(date_src)):
                g.add_edges(node_id[date_src[i]], node_id[date_dst[i]])
                g.add_edges(node_id[date_dst[i]], node_id[date_src[i]])
    
            
            
            self.graphs.append(g)
        
            if label_dict[graph_name] == 'Completion':
                self.labels.append([1,0])
            else:
                self.labels.append([0,1])
            
            
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
    

    def save(self):
        # save graphs and labels
        graph_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        #info_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph_info.pkl')
        #save_info(info_path, {'num_classes': self.num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        #info_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph_info.pkl')
        #self.num_classes = load_info(info_path)['num_classes']
        