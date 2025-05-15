FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get update && apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install matplotlib
RUN pip install pandas

RUN pip install memory_profiler==0.61.0
RUN pip install numpy==1.24.2
RUN pip install scikit_learn
RUN pip install scipy==1.10.1
RUN pip install seaborn==0.11.1
RUN pip install torch==2.0.0
RUN pip install torch_geometric==2.2.0
RUN pip install torch_scatter==2.1.1+pt20cu117 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
RUN pip install torch_sparse==0.6.17+pt20cu117 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
RUN pip install tqdm==4.55.0
RUN pip install optuna
RUN pip install gdown

WORKDIR /home/
