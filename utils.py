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
    plt.savefig('./images/' + model + '_loss_' + '_epoch_' + str(num_epochs) + '_layers_' + str(num_layers) + '_lr_' + str(lr) + '.png')
    

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

def one_hot_encoding(n, n_to_index):
    one_hot_vector = [0]*(len(n_to_index))
    index = n_to_index[n]
    one_hot_vector[index] = 1
    return one_hot_vector

def oulad_load():
    path = 'data/oulad'

    courses = pd.read_csv(os.path.join(path, 'courses.csv'))
    vle = pd.read_csv(os.path.join(path, 'vle.csv'))
    studentVle = pd.read_csv(os.path.join(path, 'studentVle.csv'))
    coursstudentRegistrationes = pd.read_csv(os.path.join(path, 'studentRegistration.csv'))
    studentAssessment = pd.read_csv(os.path.join(path, 'studentAssessment.csv'))
    studentInfo = pd.read_csv(os.path.join(path, 'studentInfo.csv'))
    assessments = pd.read_csv(os.path.join(path, 'assessments.csv'))
    
    return courses, vle, studentVle, studentRegistration, studentAssessment, studentInfo, assessments