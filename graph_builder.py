import pandas as pd
import numpy as np
import dgl
from dgl.data import DGLDataset
import torch
from tqdm import tqdm
import os
from dgl import save_graphs, load_graphs
from sklearn.model_selection import train_test_split
from utils import oulad_load, one_hot_encoding, flat_cols, find_next_learning, count_activity, add_learning_sequence
        
class Oulad_main_Dataset(DGLDataset):
    def __init__(self, days, split_ratio):
        self.days = days
        self.split_ratio = split_ratio
        super().__init__(name='oulad')
        
    def process(self):
        # graph_path = os.path.join('data/oulad', 'Oulad_main_Dataset_graph.bin')
        # self.graphs = load_graphs(graph_path)
        # return
        
        courses, vle, studentVle, studentRegistration, studentAssessment, studentInfo, assessments = oulad_load()

        # Completion == 1, Dropout == 0
        studentInfo['completion_status'] = list(map(lambda x: 1 if (x == 'Pass') or (x == 'Distinction') else 0, studentInfo['final_result']))
        studentInfo['course_name'] = studentInfo['code_module'] + '_' + studentInfo['code_presentation']
        vle['course_name'] = vle['code_module'] + '_' + vle['code_presentation']
        studentVle['course_name'] = studentVle['code_module'] + '_' + studentVle['code_presentation']
        studentVle['graph_name'] = studentVle['code_module'] + '_' + studentVle['code_presentation'] + '_' + studentVle['id_student'].astype(str)

        # 첫번째 인터랙션 날을 기준으로 이후 30일간으로 맞추기 위해 첫번째 인터랙션 날을 기준으로 뺄셈
        student_first_date = studentVle[['graph_name', 'date']].groupby('graph_name').agg(['min']).pipe(flat_cols)
        studentVle = pd.merge(studentVle, student_first_date, how='left', on='graph_name')
        studentVle['new_date'] = studentVle['date'] - studentVle['date / min']

        course_learning_sequence = {}
        learning_sequence = []

        studentVle = studentVle[studentVle['new_date'] < self.days]

        for course in tqdm(list(set(vle['course_name']))):
            target_course = studentVle[studentVle['course_name'] == course]

            target_course_student = set(target_course['id_student'].to_list())
            target_course_activity = set(target_course['id_site'].to_list())

            student_learning = {}
            first_learning = []

            # 학생들 별로 learning sequence 담기
            for student in target_course_student:
                learing_list = target_course[target_course['id_student'] == student]['id_site'].to_list()
                student_learning[student] = learing_list
                first_learning.append(learing_list[0])

            # 첫번째 시작 activity 계산
            num_activity = count_activity(first_learning)
            target_activity = num_activity[0][0]
            learning_sequence.append(target_activity)

            for i in range(len(target_course_activity)):
                bf_ls_len = len(learning_sequence)
                next_learning = find_next_learning(target_course_student, target_activity, student_learning)
                num_next_learning = count_activity(next_learning)
                learning_sequence, target_activity = add_learning_sequence(num_next_learning, learning_sequence)
                if bf_ls_len == len(learning_sequence):
                    break
            
            course_learning_sequence[course] = learning_sequence
            
        # Graph build 순서
        # course node 만들기 -> activity node 만들기 -> activity node와 course node 사이에 edge 생성
        # activity node 사이에 edge만들기
        # student node만들기 -> activity node와 edge 생성 -> course node와 edge 생성
        studentVle = studentVle[studentVle['date'] >= 0]
        group_studentvle = studentVle.groupby(['id_student', 'id_site']).sum().reset_index()

        node_set = {}
        index = 0
        course_node = list(set(vle['course_name']))
        activity_node = list(set(vle['id_site']))
        student_node = list(set(studentInfo['id_student']))
        for i in course_node + activity_node + student_node:
            node_set[i] = index
            index += 1
            
        edge_feature = []
        # course node, activity node 만들기 -> activity node와 course node 사이에 edge 생성 가중치 1
        src_node, dst_node = [], []
        for i in range(len(vle)):
            #src_node.append(node_set[vle['course_name'].to_numpy()[i]])
            #dst_node.append(node_set[vle['id_site'].to_numpy()[i]])
            src_node.append(node_set[vle['course_name'][i]])
            dst_node.append(node_set[vle['id_site'][i]])
            edge_feature.append(1)
            
            
        # student node만들기 -> activity node와 edge 생성 click_sum z-score
        for i in range(len(group_studentvle)):
            src_node.append(node_set[group_studentvle['id_student'][i]])
            dst_node.append(node_set[group_studentvle['id_site'][i]])
            edge_feature.append(group_studentvle['sum_click'][i])
            #edge_feature.append(group_studentvle[(group_studentvle['id_student'] == studentVle['id_student'][i]) & (group_studentvle['id_site'] == studentVle['id_site'][i])]['sum_click'].values[0])
            
        # course node와 edge 생성 completion한 사람만 edge 생성 가중치 1
        for i in range(len(studentInfo)):
            if studentInfo['completion_status'][i] == 'Completion':
                src_node.append(node_set[studentInfo['id_student'][i]])
                dst_node.append(node_set[studentInfo['course_name'][i]])
                edge_feature.append(1)
                
        # activity 사이에 edge 생성
        for course in list(set(vle['course_name'])):
            for i in range(len(course_learning_sequence[course])):
                src_node.append(node_set[course])
                dst_node.append(node_set[course_learning_sequence[course][i]])
                edge_feature.append(1)

        graph_src_node = src_node + dst_node
        graph_dst_node = dst_node + src_node
        g = dgl.graph((graph_src_node, graph_dst_node))

        g.edata['edge_feature'] = torch.FloatTensor(edge_feature + edge_feature)

        node_feature = []

        # make one-hot vector using activity_type
        activity_onehot_list = list(vle['activity_type'].unique())
        activity_zero_list = [0 for i in range(len(activity_onehot_list))]
        activity_to_index = {n : index for index, n in enumerate(activity_onehot_list)}
        activity_type = []
        for activity_id in activity_node:
            activity_type.append(vle[vle['id_site'] == activity_id]['activity_type'].values[0])
            
        # activity one-hot
        activity_node_feature = []
        for activity_type in activity_type:
            activity_node_feature.append(one_hot_encoding(activity_type, activity_to_index))
            
        # date one-hot
        date_feature = []
        for student in student_node:
            base_date_feature = [0]*self.days
            student_date = list(set(studentVle[studentVle['id_student'] == student]['new_date']))
            for date in student_date:
                base_date_feature[date] = 1
            date_feature.append(base_date_feature)
        base_date_feature = [0]*self.days

        for i in range(len(course_node)):
            node_feature.append([0,0,0] + activity_zero_list + base_date_feature)

        for i in range(len(activity_node)):
            node_feature.append([1,0,0] + activity_node_feature[i] + base_date_feature)
            
        for i in range(len(student_node)):
            node_feature.append([0,1,0] + activity_zero_list + date_feature[i])
            
        g.ndata['feature'] = torch.FloatTensor(node_feature)
        
        

        course_student = {}
        pos_student = studentInfo[studentInfo['completion_status'] == 1] # Completion 15385
        neg_student = studentInfo[studentInfo['completion_status'] == 0] # Dropout 17208
        train_pos, train_neg, test_pos, test_neg = {'src':[], 'dst':[]}, {'src':[], 'dst':[]}, {'src':[], 'dst':[]}, {'src':[], 'dst':[]}
        for course in course_node:
            pos_src_list = []
            pos_dst_list = []
            
            neg_src_list = []
            neg_dst_list = []
            
            for i in pos_student[pos_student['course_name'] == course]['id_student'].tolist():
                pos_src_list.append(node_set[i])
                pos_dst_list.append(node_set[course])
                
            for i in neg_student[neg_student['course_name'] == course]['id_student'].tolist():
                neg_src_list.append(node_set[i])
                neg_dst_list.append(node_set[course])
            
            train_pos_src, test_pos_src, train_pos_dst, test_pos_dst = train_test_split(pos_src_list, pos_dst_list, test_size=1-self.split_ratio, train_size=self.split_ratio, random_state=32)
            train_neg_src, test_neg_src, train_neg_dst, test_neg_dst = train_test_split(neg_src_list, neg_dst_list, test_size=1-self.split_ratio, train_size=self.split_ratio, random_state=32)

            train_pos['src'] = train_pos['src'] + train_pos_src
            train_pos['dst'] = train_pos['dst'] + train_pos_dst
            train_neg['src'] = train_neg['src'] + train_neg_src
            train_neg['dst'] = train_neg['dst'] + train_neg_dst
            test_pos['src'] = test_pos['src'] + test_pos_src
            test_pos['dst'] = test_pos['dst'] + test_pos_dst
            test_neg['src'] = test_neg['src'] + test_neg_src
            test_neg['dst'] = test_neg['dst'] + test_neg_dst
            
        num_node = len(course_node + activity_node + student_node)
        train_pos_g = dgl.graph((train_pos['src'], train_pos['dst']), num_nodes=num_node)
        train_neg_g = dgl.graph((train_neg['src'], train_neg['dst']), num_nodes=num_node)
        test_pos_g = dgl.graph((test_pos['src'], test_pos['dst']), num_nodes=num_node)
        test_neg_g = dgl.graph((test_neg['src'], test_neg['dst']), num_nodes=num_node)

        self.graphs = []
        self.graphs.append(g)
        self.graphs.append(train_pos_g)
        self.graphs.append(train_neg_g)
        self.graphs.append(test_pos_g)
        self.graphs.append(test_neg_g)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)
    

    def save(self):
        # save graphs and labels
        graph_path = os.path.join('data/oulad', 'Oulad_main_Dataset_graph.bin')
        save_graphs(graph_path, self.graphs)
        # save other information in python dict
        #info_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph_info.pkl')
        #save_info(info_path, {'num_classes': self.num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join('data/oulad', 'Oulad_main_Dataset_graph.bin')
        self.graphs = load_graphs(graph_path)
        #info_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph_info.pkl')
        #self.num_classes = load_info(info_path)['num_classes']
        