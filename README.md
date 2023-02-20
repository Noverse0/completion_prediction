# online_completion_prediction

### 작업목록
click_sum을 edge feature로 이용한 gcn baseline model 구현 완료 -> gcn으로 명칭

[todo]  
우리 Model 구현중


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
OULAD: completion_prediction/data/archive/~

Download  
Naver: ~  
OULAD: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad?select=vle.csv

### Run
```
python3 main.py -d [data name oulad or ]
```
