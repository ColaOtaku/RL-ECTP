# RL-ECTP
Official implementation of RL-ECTP: Towards An Efficient and Effective Framework for Traffic Streaming Prediction on Traffic Sensor Networks

## 1.Repository Structure
```
code/
├── data/
│ ├── ca/                  # data pre-process code and data storage for CA
│ └── chengdu/             # data pre-process code and data storage for CD
├── experiments/rl_ectp/   # main script to run the experiments
│ ├── CA/                  # model weights for evaluation(CA)
│ └── chengdu/             # model weights for evaluation(CD)
├── src/      
│ ├── base/                # base classes for engines and models
│ ├── engines/             # The script for training and evaluation
│ ├── models/              # Scripts for pytorch implementation of models
│ └── utils/               # some utils functions, graph_algo.py is for graph partition and tree construction
├── README.md
└── LICENSE
```

## 2.Requirements
Versions of python==3.10, pytorch==2.3.1, numpy==1.24.4, pandas==2.0.3, metis==0.2a5

## 3.Data
  We use CA and CD as our experiment datasets. References are https://www.kaggle.com/datasets/liuxu77/largest and https://outreach.didichuxing.com/research/opendata/ from papers 1. LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting and 2. An Effective Joint Prediction Model for Travel Demands and Traffic Flows. 

1. For CA, use `data/ca/process_ca_his.py` to get `ca_his_2019.h5` and use `data/ca/generate_data_for_training_ca.py` to get the processed CA data under `data/ca/2019/`.
2. For CD, use `data_analytic1.ipynb` and `data_analytic2.ipynb` to get `data/chengdu/raw/cnt_300.npy` and the adj matrix, then use `data/ca/generate_data_for_training_didi.py` to get the processed CD data under `data/chengdu/2016/`.

<p align="center">
  <img src="https://github.com/ColaOtaku/RL-ECTP/blob/main/images/table3.jpg" alt="data">
</p>

## 4.Experiment
Model training and evaluation can be done through scripts at `experiments/rl_ectp/run.sh`, and we also provide the model weights in `experiments/rl_ectp/CA/` and `experiments/rl_ectp/chengdu/` to reproduce the result.
<p align="center">
  <img src="https://github.com/ColaOtaku/RL-ECTP/blob/main/images/table4.jpg" alt="result1">
</p>

<p align="center">
  <img src="https://github.com/ColaOtaku/RL-ECTP/blob/main/images/table5.jpg" alt="result2">
</p>

## 5.Acknowledgments
We gratefully acknowledge the contribution of [LargeST](https://github.com/liuxu77/LargeST), whose benchmark code was used as a reference.
