default: build

help:
	@echo 'Management commands for igmc:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the airflow_pipeline project project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t completion_prediction --platform linux/amd64

run:
	@echo "Booting up Docker Container"
	@docker run -it --ipc=host --name completion_prediction -v `pwd`:/workspace/completion_prediction completion_prediction:latest /bin/bash

up: build run

rm: 
	@docker rm completion_prediction

stop:
	@docker stop completion_prediction

reset: stop rm