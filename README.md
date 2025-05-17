# FairGraphBase

FairGraphBase is a benchmarking framework for fairness-aware graph neural networks.
This supports (1) in-processing and pre-processing methods, (2) their combinations, and (3) different early stopping conditions.

<div  align="center">    
<img src="./docs/musubi.png" width=500  alt="musubi" align=center />
</div>

## Installation
The FairGraphBase codebase uses the following dependencies:
- python=3.10.13
- CUDA11.7
- pytorch=2.0.0
- torch_geometric=2.2.0

## Supported Models for group fairness
- Inprocessing<br>
Vanilla, FairGNN ([\[WSDM 2021\]](https://arxiv.org/pdf/2009.01454.pdf)), NIFTY ([\[UAI 2021\]](https://arxiv.org/pdf/2102.13186.pdf)), FairSIN ([\[AAAI 2024\]](https://arxiv.org/pdf/2403.12474))
- Preprocessing<br>
Undersampling, BIND ([\[AAAI 2022\]](https://arxiv.org/pdf/2211.14383))
- Encoder<br> 
GCN, GAT, SAGE, H2GCN

## Supported graph datasets
We support the following graphs. In our experiments, German, Google, AMiner-S, and AMiner-L were not suitable to evaluate the performance (we will investigate further).


|                |               #Nodes               |   #Edges   | #Features |
| :------------: | :--------------------------------: | :--------: | :-------: |
|   **German**   |               1,000                |   24,970   |    27     |
|   **Bail**   |               18,876               |  642,616   |    18     |
|   **Credit**   |               30,000               |  304,754   |    13     |
|  **Pokec-n**   |               7,659               |  36,827   |    277    |
|  **Pokec-z**   |               6,185               |  48,759   |    266    |
|  **Pokec-n-Large**   |               66,569               |  1,100,663   |    266    |
|  **Pokec-z-Large**   |               67,796               |  1,303,712   |    277    |
|  **Google**+   |              290,468               |   3,601    |   2,532   |
|  **AMiner-S**  |               39,424               |   52,460   |   5,694   |
|  **AMiner-L**  |              129,726               |  591,039   |   5,694   |
|  **YAGO**  |              127,752               |  349,942   |   114   |
|  **DBpedia**   |              152,035               |   426,851    |   478   |
|  **Wikidata**  |               1,108,347               |   4,857,725   |   415   |


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

## Future extension
- Add other methods, e.g., fairdrop
- Add other fairness metrics
