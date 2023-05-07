FROM public.ecr.aws/lambda/python:3.8

RUN yum -y install gcc
RUN yum -y install libmpc-devel

COPY requirements.txt  .

RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY . ${LAMBDA_TASK_ROOT}
RUN python3.8 download.py

CMD ["app.lambda_handler"]