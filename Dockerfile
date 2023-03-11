FROM python:3.11
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
COPY requirements.txt .
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get -y install gcc
RUN pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
RUN pip install --trusted-host pypi.python.org -r requirements.txt

WORKDIR /workspace/completion_prediction

#COPY . .
#CMD [ "python3", "-W ignore" ,"main.py", "-d", "oulad", "--gpu", "0", "--num_layers", "2"]
