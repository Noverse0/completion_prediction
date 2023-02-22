FROM python:3.11
COPY requirements.txt .
RUN apt-get update
RUN apt-get -y install gcc
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . .
CMD [ "python3", "-W ignore" ,"main.py", "-d", "oulad", "--gpu", "0", "--num_layers", "2"]
