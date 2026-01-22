## Usage
1. first install smoothquant:
```bash
conda create -n smoothquant python=3.8
conda activate smoothquant
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.36.0 accelerate datasets zstandard

python setup.py install
```

2. To run the test you will need a huggingface token
```bash
python layer_sq/benchmark_layerSQ.py
```