# samul
## Semantic Averaging with Multiple Languages

This repository contains application code to prepare, process and output a PhD experiment in machine learning, LLMs, Sparse Autoencoders and Mechanistic Intepretability. This repository is tailored to a specific configuration of inputs and ground-truth mappings (e.g. from the ontology matching task here: https://oaei.ontologymatching.org/2024/conference/index.html and other corpora), but it can be amended to include any sources.

More detailed runtime configuration is described in the comments in samul.py.

The experiment takes a text file as input that contains a set of concept references and fragments of English language. A mapping file is required that provides ground truth similarity score between pairs of texts. This code proceeds to do the following:

<ol>
  <li>For each text file, translate the English version into a non-English version (based on the config settings) and store these new translated files,</li>
          <li>Both non-English and English texts are processed via LLM Sparse Autoencoders to extract a tensor representation of the feature activations,</li>
<li>For each concept, a tensor average is produced over the English and non-English version (e.g. en/fr and/or en/de),</li>
<li>For each pair of mappings the cosine similarity is computed between the English-only tensor and each average tensor,</li>
<li>The cosine similarities are correlated with the ground truth similarity for each pairing,</li>
<li>A single number is output for each corpus and translated language that measures the effectiveness of the semantic representations from the Sparse Autoencoder and the ground truth similarities from the source corpus.</li>
</ol>

It has been established in previous experiments that the average of the English and non-English text tensor representations is more strongly correlated to the ground truth semantic similarity.

## 🚀 Usage

### Requirements

1. This experiment has been run successfully on Mac OS running Sequioa 15.3.2

2. Python 3.10.19 and the following libraries (also listed in requirements.txt):

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

4. An account and credentials to use the GemmaScope SAE via the HuggingFace implementation (details can be found here: https://huggingface.co/google/gemma-scope). Once a token has been created on the HuggingFace account, run the setup script:

        python setup.py

   This will enable the entry and local storage of the token so that the weights can be downloaded locally.


### Execution

1. Change the config.json file to reflect the various local settings needed to run the experiment, including the experiment folder location, file names etc

2. Start the process with:

        python samul.py









