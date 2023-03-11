# completion_prediction

### 작업목록

oulad dataset을 이용하여 GCN을 이용하여 mainmodel 만듬

[todo]  
우리 Model mooc dataset으로
Window를 만들어 Dropout predcition으로 다시 문제정의

### About

We are conducting research to predict completion rates.  
Our method is graph-based and focuses on student interactions.  
Also, it adds novelty by creating a meta path between activity and assessment.  
The number of layers is also taken into account so that messages can be delivered to all nodes in the graph.

### Requirements

```
...
```

Install [DGL](https://www.dgl.ai/pages/start.html)

### Data

We use the Naver edwith dataset and the Open University Learning Analytics dataset (OULAD).  
Data is too large to upload.
Please set the data like the path below.

Naver: completion_prediction/data/NAVER_Connect_Edwith_Dataset_v2.0/~  
OULAD: completion_prediction/data/oulad/~
Mooc: completion_prediction/data/mooc/~

Download  
Naver: ~  
OULAD: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad?select=vle.csv
Mooc: http://moocdata.cn/data/user-activity

### Run

docker build & run method

```
make up
```

CMD Example

```
python main.py -d oulad --gpu 0 --num_layer 2 --threshold 0.4 --num_epochs 50 --lr 0.01
```
