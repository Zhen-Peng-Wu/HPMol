
## Requirement

```shell
accelerate == 1.0.1
bitsandbytes == 0.44.1
huggingface-hub == 0.25.2
protobuf == 3.20.0
rdkit == 2023.3.2
tokenizers == 0.20.1
transformers == 4.45.2
torch == 2.0.1+cu117
torch-geometric == 2.6.0
```



## Installation

```shell
conda create -n HPMol python==3.9 -y
conda activate HPMol
pip install protobuf==3.20.0
pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.45.2
pip install sentencepiece==0.2.0
pip install accelerate==1.0.1
pip install bitsandbytes==0.44.1
pip install rdkit==2023.3.2
pip install torch-geometric==2.6.0
pip install huggingface-hub==0.25.2
pip install tokenizers==0.20.1
pip install scikit-learn==1.6.1
pip install deepchem==2.8.0
pip install numpy==1.26.4
```

## Dataset Preparation

```shell
cd HPMol-master/
```

download the datasets from Google Drive:
```shell
https://drive.google.com/open?id=1xI9u0w-NF8brPJRLlQijCyvHEiL-QZC5
```

then unzip the file
```shell
unzip HPMol_dataset.zip
```


## Quick Start

**A quick start example is given by:**
```shell
$ python main_gnn_classification.py --task bace
```

**If user wants to re-obtain the LLM embedding derived from SMILEs:**
```shell
$ python emb_generate.py --task bace
```

