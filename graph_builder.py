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
        
class Oulad_Dataset(DGLDataset):
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
        assessments['course_name'] = assessments['code_module'] + '_' + assessments['code_presentation']
        studentVle['course_name'] = studentVle['code_module'] + '_' + studentVle['code_presentation']
        studentVle['graph_name'] = studentVle['code_module'] + '_' + studentVle['code_presentation'] + '_' + studentVle['id_student'].astype(str)

        # 첫번째 인터랙션 날을 기준으로 이후 30일간으로 맞추기 위해 첫번째 인터랙션 날을 기준으로 뺄셈
        student_first_date = studentVle[['graph_name', 'date']].groupby('graph_name').agg(['min']).pipe(flat_cols)
        studentVle = pd.merge(studentVle, student_first_date, how='left', on='graph_name')
        studentVle['new_date'] = studentVle['date'] - studentVle['date / min']
        studentVle = studentVle[studentVle['new_date'] < self.days]
        
        studentAssessment = pd.merge(studentAssessment, assessments, how='left', on='id_assessment')
        studentAssessment['graph_name'] = studentAssessment['code_module'] + '_' + studentAssessment['code_presentation'] + '_' + studentAssessment['id_student'].astype(str)
        studentAssessment = pd.merge(studentAssessment, student_first_date, how='left', on='graph_name')
        studentAssessment['new_date'] = studentAssessment['date'] - studentAssessment['date / min']
        studentAssessment = studentAssessment[studentAssessment['new_date'] < self.days]
        
        
        #----------------------Cousre Squence 만들기-------------------
        course_learning_sequence = {}
        learning_sequence = []
        
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
            
        #-----------------Graph building--------------------
        group_studentvle = studentVle.groupby(['id_student', 'id_site']).sum().reset_index()
        group_studentassessment = studentAssessment.groupby(['id_student', 'id_assessment']).last().reset_index()

        node_set = {}
        index = 0
        course_node = list(set(studentVle['course_name']))
        activity_node = list(set(studentVle['id_site']))
        assessment_node = list(set(studentAssessment['id_assessment']))
        student_node = list(set(studentVle['id_student']))
        for i in course_node + activity_node + assessment_node + student_node:
            node_set[i] = index
            index += 1
            
        studentVle = studentVle.reset_index(drop=True)
        studentAssessment = studentAssessment.reset_index(drop=True)
        group_studentvle = group_studentvle.reset_index(drop=True)
        group_studentassessment = group_studentassessment.reset_index(drop=True)
        studentInfo = studentInfo.reset_index(drop=True)

        edge_feature = []
        src_node, dst_node = [], []

        # Course Node와 Activity Node 만들기
        for i in range(len(studentVle)):
            src_node.append(node_set[studentVle['course_name'][i]])
            dst_node.append(node_set[studentVle['id_site'][i]])
            edge_feature.append(1)
            
        # Course Node와 Assessment Node 만들기
        for i in range(len(studentAssessment)):
            src_node.append(node_set[studentAssessment['course_name'][i]])
            dst_node.append(node_set[studentAssessment['id_assessment'][i]])
            edge_feature.append(1)
            
        # student node와 Activity Node 만들기
        for i in range(len(group_studentvle)):
            src_node.append(node_set[group_studentvle['id_student'][i]])
            dst_node.append(node_set[group_studentvle['id_site'][i]])
            edge_feature.append(group_studentvle['sum_click'][i])
            
        # student node와 Assessment Node 만들기
        for i in range(len(group_studentassessment)):
            src_node.append(node_set[group_studentassessment['id_student'][i]])
            dst_node.append(node_set[group_studentassessment['id_assessment'][i]])
            #edge_feature.append(group_studentassessment['score'][i])
            edge_feature.append(1)
            
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
        assessment_onehot_list = list(assessments['assessment_type'].unique())
        activity_zero_list = [0 for i in range(len(activity_onehot_list))]
        assessment_zero_list = [0 for i in range(len(assessment_onehot_list))]
        activity_to_index = {n : index for index, n in enumerate(activity_onehot_list)}
        assessment_to_index = {n : index for index, n in enumerate(assessment_onehot_list)}
        activity_type, assessment_type = [], []
        for activity_id in activity_node:
            activity_type.append(vle[vle['id_site'] == activity_id]['activity_type'].values[0])
        for assessment_id in assessment_node:
            assessment_type.append(assessments[assessments['id_assessment'] == assessment_id]['assessment_type'].values[0])
            
        # activity one-hot
        activity_node_feature, assessment_node_feature = [], []
        for activity_type in activity_type:
            activity_node_feature.append(one_hot_encoding(activity_type, activity_to_index))
        for assessment_type in assessment_type:
            assessment_node_feature.append(one_hot_encoding(assessment_type, assessment_to_index))
            
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
            node_feature.append([0,0,0] + activity_zero_list + assessment_zero_list + base_date_feature)

        for i in range(len(activity_node)):
            node_feature.append([1,0,0] + activity_node_feature[i] + assessment_zero_list + base_date_feature)
            
        for i in range(len(assessment_node)):
            node_feature.append([0,1,0] + activity_zero_list + assessment_node_feature[i] + base_date_feature)
            
        for i in range(len(student_node)):
            node_feature.append([0,0,1] + activity_zero_list + assessment_zero_list + date_feature[i])
            
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
                if i in list(node_set.keys()):
                    pos_src_list.append(node_set[i])
                    pos_dst_list.append(node_set[course])
                
            for i in neg_student[neg_student['course_name'] == course]['id_student'].tolist():
                if i in list(node_set.keys()):
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
            
        num_node = len(course_node + activity_node + assessment_node + student_node)
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
        