from huggingface_hub import login
import os


# ============================= create source data folder =============================
# create a folder called source_data. this is where the raw text data will be stored and 
# any translated versions too.

datasourcefolder = "source_data"
if not os.path.exists(datasourcefolder):
        os.makedirs(datasourcefolder)


# ============================= embed hugging face token in local environment =============================
# the login function enables a access token to be locally stored which will allow downloads of the gemma scope weights

login()
