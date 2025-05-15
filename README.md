# FairGraphBase

FairGraphBase is a benchmarking framework for fairness-aware graph neural networks.

<div  align="center">    
<img src="./docs/musubi.png" width = "600" height = "200" alt="musubi" align=center />
</div>

## Installation
The FairGraphBase codebase uses the following dependencies:
- python=3.10.13
- CUDA11.7
- pytorch=2.0.0
- torch_geometric=2.2.0

## Supported Models
- Inprocessing<br>
Vanilla, FairGNN, NIFTY, FairSIN
- Preprocessing<br>
Undersampling, BIND
- Encoder<br> 
GCN, GAT, SAGE, H2GCN

## Argument options
- dataset : dataset name. credit/bail/german/pokec_z/pokec_n/pokec_z_large/pokec_n_large/google/aminer_l/aminer_s/wikidata/dbpedia/yago (default=pokec_n)
- inprocessing : inprocessing method. vanilla/fairgnn/nifty/fairsin（default=vanilla）
- preprocessing : preprocessing method. bind/undersampling/None（default=None）
- encoder : backbone model. gcn/gat/sage/h2gcn
- optimize : Run optuna if specified
- trainsize: Train data ratio (default=0.6)
- valsize: Validation data ratio  (default=0.2)
- epochs : the number of epoch (deault=2000, if inprocessing is fairsin, default=50)
- optrials : Specify the number of optuna attempts（default=20）
- metrics : Early stopping condition. acc/f1/alpha（defalt=acc）
- alpha : Alpha value when alpha is specified in metrics（default=0.5）
- seed : random seed (default=42)
- runs : Number of trials performed（default=5）
- device : Device number when using GPU（default='cuda:0'）

## Example of command execution：
```
python main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
```

## Output
CSV files are generated at output directory

## Hardware
NVIDIA Quadro RTX 8000
