# RL-ECTP
Official implementation of RL-ECTP: Towards An Efficient and Effective Framework for Traffic Streaming Prediction on Traffic Sensor Networks

## Repository Structure
code/
├── data/
│ ├── ca/                  # data pre-process code and data storage for CA
│ └── chengdu/             # data pre-process code and data storage for CD
├── experiments/rl_ectp/   # main script to run the experiments
├── src/      
│ ├── base/                # base classes for engines and models
│ ├── engines/             # The script for training and evaluation
│ ├── models/              # Scripts for pytorch implementation of models
│ └── utils/               # some utils functions, graph_algo.py is for graph partition and tree construction
├── README.md
└── LICENSE

## Data
Data references are https://www.kaggle.com/datasets/liuxu77/largest and https://outreach.didichuxing.com/research/opendata/ from papers ‘LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting’ and ‘An Effective Joint Prediction Model for Travel Demands and Traffic Flows’. 

1. For CA, use data/ca/process_ca_his.py to get ca_his_2019.h5 and use data/ca/generate_data_for_training_ca.py to get the processed CA data under data/ca/2019/. 2. For CD, use data_analytic1.ipynb and data_analytic2.ipynb to get data/chengdu/raw/cnt_300.npy and the adj matrix, then use data/ca/generate_data_for_training_didi.py to get the processed CD data under data/chengdu/2016/.

table3

## Experiment
Model training and evaluation can be done through scripts at experiments/rl_ectp/run.sh, and we provide the model weights in experiments/rl_ectp/CA/ and experiments/rl_ectp/chengdu/ to reproduce the result.

table 4
table 5
