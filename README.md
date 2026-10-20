# samul
## Semantic Averaging with Multiple Languages

This repository contains application code to prepare, process and output a PhD experiment in machine learning, LLMs, Sparse Autoencoders and Mechanistic Intepretability. This repository is tailored to a specific configuration of inputs and ground-truth mappings (taken from the ontology matching task here: https://oaei.ontologymatching.org/2024/conference/index.html and other corpora), but it can be amended to include any sources.

More detailed runtime configuration is described in the comments in samul.py.


## 🚀 Usage

### Requirements

1: This experiment has been run successfully on Mac OS running Sequioa 15.3.2

2: Python 3.10.19 and the following libraries (also listed in requirements.txt):

    transformer-lens==2.17.0
    transformers
    huggingface_hub
    torch==2.10.0
    numpy==1.26.4
    pandas==2.3.3
    scikit-learn==1.7.2
    scipy==1.15.3
    safetensors
    blobfile
    googletrans==4.0.2
    pyarrow<21

3. Run these commands to set up the environment libraries:

    pip install -r requirements.txt
    pip install --no-deps "sparse_autoencoder @ git+https://github.com/openai/sparse_autoencoder.git@4965b941e9eb590b00b253a2c406db1e1b193942"




