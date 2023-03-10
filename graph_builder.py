import pandas as pd
import numpy as np
import dgl
from dgl.data import DGLDataset
import torch
from tqdm import tqdm
import os
from dgl import save_graphs, load_graphs
from sklearn.model_selection import train_test_split

# learning_sequence에 이미 들어 있는 activity를 제외하고 다음 activity를 추가함
def add_learning_sequence(num_next_learning, learning_sequence):
    for i in range(len(num_next_learning)):
        if num_next_learning[i][0] in learning_sequence:
            pass
        else:
            learning_sequence.append(num_next_learning[i][0])
            break
    return learning_sequence, num_next_learning[i][0]

# list에 있는 activity들의 개수를 count함
def count_activity(first_learning):
    num_activity = {}
    for activity in list(set(first_learning)):
        num_activity[activity] = first_learning.count(activity)
    num_activity = sorted(num_activity.items(),reverse=True, key=lambda x:x[1])
    return num_activity

# next_learning이라는 리스트에 우리의 target_activity 다음에 오는 activity들을 추가함
def find_next_learning(student_list, target_activity, student_learning):
    next_learning = []
    for student in student_list:
        first_activity_index = [i for i in range(len(student_learning[student])) if target_activity == student_learning[student][i]]
        for index in first_activity_index:
            if index+1 >= len(student_learning[student]):
                pass
            else:
                next_learning.append(student_learning[student][index+1])
    return next_learning

# MultiIndex 컬럼을 평탄화 하는 함수
def flat_cols(df):
    df.columns = [' / '.join(x) for x in df.columns.to_flat_index()]
    return df

def oulad_load():
    courses = pd.read_csv('data/archive/courses.csv')
    vle = pd.read_csv('data/archive/vle.csv')
    studentVle = pd.read_csv('data/archive/studentVle.csv')
    studentRegistration = pd.read_csv('data/archive/studentRegistration.csv')
    studentAssessment = pd.read_csv('data/archive/studentAssessment.csv')
    studentInfo = pd.read_csv('data/archive/studentInfo.csv')
    assessments = pd.read_csv('data/archive/assessments.csv')
    
    return courses, vle, studentVle, studentRegistration, studentAssessment, studentInfo, assessments

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
        # graph_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph.bin')
        # self.graphs, label_dict = load_graphs(graph_path)
        # self.labels = label_dict['labels']
        # return
        
        courses, vle, studentVle, studentRegistration, studentAssessment, studentInfo, assessments = oulad_load()

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
        #activity = activity[activity['date'] < self.days]
        
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

            src_node, dst_node = [], []
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
                if node < self.days: # date node
                    date_node_feature.append(edges_of_id['date_node_feature'].to_numpy()[[np.where(edges_of_id['new_date'].to_numpy() == node)[0][0]]][0])
                    activity_node_feature.append(activity_zero_list)
                    assessment_node_feature.append(assessment_zero_list)
                    node_type.append([0,0,1])
                else:
                    date_node_feature.append([0 for i in range(self.days)])
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

            
            # edge date with date
            date_src = list(set(edges_of_id['new_date']))[:-1]
            date_dst = list(set(edges_of_id['new_date']))[1:]
            
            for i in range(len(date_src)):
                g.add_edges(node_id[date_src[i]], node_id[date_dst[i]])
                g.add_edges(node_id[date_dst[i]], node_id[date_src[i]])

            date_edge_feature = torch.FloatTensor([1 for i in range(len(list(set(edges_of_id['new_date']))[1:]))])
            g.edata['edge_feature'] = torch.FloatTensor(list(edges_of_id['sum_click']) + list(edges_of_id['sum_click']) + [1 for i in range(len(list(set(edges_of_id['new_date']))[1:]))] + [1 for i in range(len(list(set(edges_of_id['new_date']))[1:]))])
            
            #print(g.edata['edge_feature'])
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
        
        
class Oulad_main_Dataset(DGLDataset):
    def __init__(self, days, split_ratio):
        self.days = days
        self.split_ratio = split_ratio
        super().__init__(name='oulad')
        
    def process(self):
        # graph_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph.bin')
        # self.graphs, label_dict = load_graphs(graph_path)
        # self.labels = label_dict['labels']
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
            
            train_pos_src, test_pos_src, train_pos_dst, test_pos_dst = train_test_split(pos_src_list, pos_dst_list, test_size=0.2, train_size=0.8, random_state=32)
            train_neg_src, test_neg_src, train_neg_dst, test_neg_dst = train_test_split(neg_src_list, neg_dst_list, test_size=0.2, train_size=0.8, random_state=32)

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

        self.graphs = g
        self.train_pos_g = train_pos_g
        self.train_neg_g = train_neg_g
        self.test_pos_g = test_pos_g
        self.test_neg_g = test_neg_g

    def __getitem__(self):
        return self.graphs, self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g

    def __len__(self):
        return len(self.graphs)
    

    def save(self):
        # save graphs and labels
        graph_path = os.path.join('data/archive', 'Oulad_main_Dataset_graph.bin')
        save_graphs(graph_path, self.graphs)
        # save other information in python dict
        #info_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph_info.pkl')
        #save_info(info_path, {'num_classes': self.num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join('data/archive', 'Oulad_main_Dataset_graph.bin')
        self.graphs = load_graphs(graph_path)
        #info_path = os.path.join('data/archive', 'oulad_baseline_gcn_graph_info.pkl')
        #self.num_classes = load_info(info_path)['num_classes']
        