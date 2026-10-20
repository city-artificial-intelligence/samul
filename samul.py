import json
from utils import *
import os
import shutil
from validation import *
from translation import *
from activations import *
from averages import *
from correlations import *
from similarities import *
from pathlib import Path


'''

This program runs a series of experiment steps to anaylyse text inputs via LLMs and Sparse Autoencoders. Guidance below:

    - Base English language is in a single file of csv format: {Concept, Text}, and should be stored in the source_data 
        folder (which is created by the setup.py script) and referenced in the config file in the data_start_file attribute
    - Ground truth file csv format is {Source File, Concept 1, Concept 2, Similarity Value} and referenced in the config file 
        in the ground_truth_file attribute
    - Language codes are taken from Google's translation service, e.g. https://developers.google.com/workspace/admin/directory/v1/languages
    - Source data filenames should not contain "-{language code}", e.g. "-fr" or "-de" in an input filename would cause errors in processing
    - The target language during development was English and is hard-coded to be this way
    - Two non-original languages are supported; the target non-English languages during development were French and Chinese
    - The development environment was MacOS 15, but any Python environment should be supported, however some minor adjustments may be 
        necessary, e.g. folder strng names and/or Torch device (MPS was used for development)
    - Huggingface usage of gemma scope requires a login and download of credentials (access token) to copy the models to a local environment. 
        This is done via the setup.py script which should be run first after cloning
    - The config attributes process_validation, process_translation etc are used to control whether these sections of the scripts execute. Therefore
        any section of the process can be switched off for whatever reason (there are dependencies though so this should be managed)
    - The type of activation calculation ('max-per-concept','mean-per-concept', 'sum-per-concept' and 'top-1-per-token') is set in 
        the config variable activation_function
    - The type of average calculations ('intersection_mean', 'full_mean', 'harmonic_mean' and 'geometric_mean') is set in 
        the config variable average_function
    - Two sparse autoencoders are supported: Gemma Scope 1 and GPT2 which require values of "google/gemma-2-2b" or "openai/gpt2" in the 
        config value "sparse_autoencoder"
    - Depending on the autoencoder chosen, some other parameters are needed, e.g. layer_sparsity_type or filerepo
    - Filerepo should take different value depending on the SAE used. For GPT2 the options are mlp_post_act, resid_delta_attn, resid_post_attn
        , resid_delta_mlp or resid_post_mlp and for Gemma Scope the value used is google/gemma-scope-2b-pt-res, although other file weights to 
        exist here: https://huggingface.co/google/gemma-scope
    - Layer numbers are 0-based, therefore to target the 12th layer in GPT2 use the value of 11 in "activations_layer_list" config item
    - If testing a small corpus, be careful not to feed a large ground truth file because there may be a mismatch between the ground truth 
        counts and the number in the activations dataset leading to a crash. The best way around this is to include only ground truth 
        values that are in the corpus
    - Each time the experiment is run it will use the id value from the config attribute experiment_id to determine which folder to use for 
        storing outputs (e.g. /ex1 or /ex3 etc). This id should be changed manually when needing to create a new experiment. Running the code 
        with the same id could result in outputs that are overwritten or corrupted. These processes are not necessarily idempotent, but manual 
        deletions can be done to re-use and save time, e.g. delete the averages folder manually but leave the previous activations folder so 
        that new averages will use the existing activations instead of creating a new experiment id and starting from scratch each time.

Code Version 1.3
Latest Change: 09/10/2026

'''


# load config
config_file = "config.json"
with open(config_file, "r") as f:   
    config = json.load(f)

process_validation = config["process_validation"]
process_translations = config["process_translations"]
process_activations = config["process_activations"]
process_averages = config["process_averages"]
process_similarities = config["process_similarities"]
process_correlations = config["process_correlations"]
experiment_folder = config["experiment_folder"]
experiment_id = config["experiment_id"]
log_file = config["log_file"]
delete_existing = config["delete_existing"]
data_source_folder = config["data_source_folder"]
corpus = config["corpus"]
data_start_file = config["data_start_file"]
ground_truth_file = config["ground_truth_file"]
languages = config["languages"]
translated_source_folder = config["translated_source_folder"] + '_' + corpus
activations_source_folder = config["activations_source_folder"] + '_' + corpus
activations_layer_list = config["activations_layer_list"]
s_autoencoder = config["sparse_autoencoder"]
filerepo = config["filerepo"]
if s_autoencoder == "google/gemma-scope-2b-pt-res":
    layer_locations = config[filerepo]
else:
    layer_locations = ""
sparsity = config["sparsity"]
layer_sparsity_type = config["layer_sparsity_type"]
concept_pos_in_file = config["concept_pos_in_file"]
averages_source_folder = config["averages_source_folder"] + '_' + corpus
multi_row_concept_match = config["multi_row_concept_match"]
similarities_source_folder = config["similarities_source_folder"] + '_' + corpus
activation_function = config["activation_function"]
average_function = config["average_function"]
ground_truth_file = experiment_folder + '/' + data_source_folder + '/' + ground_truth_file
this_exp_folder = experiment_folder + '/ex' + experiment_id
source_data_folder = experiment_folder + '/' + data_source_folder
translations_folder = experiment_folder + '/ex' + experiment_id + '/' + translated_source_folder
activations_folder = experiment_folder + '/ex' + experiment_id + '/' + activations_source_folder
averages_folder = experiment_folder + '/ex' + experiment_id + '/' + averages_source_folder
similarities_folder = experiment_folder + '/ex' + experiment_id + '/' + similarities_source_folder



# ============================= create experiment folder =============================M
if delete_existing:
    if os.path.exists(this_exp_folder):
        shutil.rmtree(this_exp_folder)
 
# create experiment folder
if not os.path.exists(this_exp_folder):
    os.makedirs(this_exp_folder)


# ============================= create log =============================
log_file = this_exp_folder + '/' + log_file
create_log('samul', log_file)


update_log(log_file, 'process_validation=' + str(process_validation))
update_log(log_file, 'process_translations=' + str(process_translations))
update_log(log_file, 'process_activations=' + str(process_activations))
update_log(log_file, 'process_averages=' + str(process_averages))
update_log(log_file, 'process_similarities=' + str(process_similarities))
update_log(log_file, 'process_correlations=' + str(process_correlations))
update_log(log_file, 'experiment_folder=' + experiment_folder)
update_log(log_file, 'experiment_id=' + str(experiment_id))
update_log(log_file, 'log_file=' + log_file)
update_log(log_file, 'delete_existing=' + str(delete_existing))
update_log(log_file, 'data_source_folder=' + data_source_folder)
update_log(log_file, 'corpus=' + corpus)
update_log(log_file, 'data_start_file=' + data_start_file)
update_log(log_file, 'ground_truth_file=' + ground_truth_file)
update_log(log_file, 'languages=' + str(languages))
update_log(log_file, 'filerepo=' + str(filerepo))
update_log(log_file, 'sparsity=' + str(sparsity))
update_log(log_file, 'layer_sparsity_type=' + str(layer_sparsity_type))
update_log(log_file, 'translated_source_folder=' + translated_source_folder) 
update_log(log_file, 'activations_source_folder=' + activations_source_folder) 
update_log(log_file, 'activations_layer_list=' + str(activations_layer_list))
update_log(log_file, 'sparse_autoencoder=' + s_autoencoder)
update_log(log_file, 'layer_locations=' + str(layer_locations))
update_log(log_file, 'concept_pos_in_file=' + str(concept_pos_in_file))
update_log(log_file, 'averages_source_folder=' + averages_source_folder)
update_log(log_file, 'multi_row_concept_match=' + str(multi_row_concept_match))
update_log(log_file, 'similarities_source_folder=' + similarities_source_folder) 
update_log(log_file, 'activation_function=' + activation_function)
update_log(log_file, 'average_function=' + average_function)
update_log(log_file, 'ground_truth_file=' + ground_truth_file) 
update_log(log_file, 'this_exp_folder=' + this_exp_folder) 
update_log(log_file, 'source_data_folder=' + source_data_folder) 
update_log(log_file, 'translations_folder=' + translations_folder) 
update_log(log_file, 'activations_folder=' + activations_folder) 
update_log(log_file, 'averages_folder=' + averages_folder) 
update_log(log_file, 'similarities_folder=' + similarities_folder) 


# ============================= validate data files =============================
if process_validation:

    thisProcess = "validation"

    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    data_source_file = source_data_folder + '/' + data_start_file
    returnVal = validate_source_file(data_source_file)
    update_log(log_file, returnVal)


    ground_truth_source_file = ground_truth_file
    returnVal = validate_ground_truth_file(ground_truth_source_file)
    update_log(log_file, returnVal)


# ============================= translations =============================
if process_translations:

    thisProcess = "translation"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)

    for lang in languages:
        new_lang_source_file = data_source_file.replace('.csv', '-' + lang + ".csv")
        nfsfp = Path(new_lang_source_file)

        if nfsfp.exists():
            msg = "Source data file for language " + lang + " already exists in " + source_data_folder
            print(msg)
            update_log(log_file, msg)
        else:
            translate_from_source('en', lang, data_source_file, source_data_folder, log_file, delete_existing)


# ============================= activations =============================
if process_activations:

    thisProcess = "activations"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    # ============================= delete/create activations folder =============================
    if delete_existing:
        if os.path.exists(activations_folder):
            shutil.rmtree(activations_folder)

    if not os.path.exists(activations_folder):
        os.makedirs(activations_folder)

    # original language file
    calculate_activations(thisProcess, log_file, s_autoencoder, activations_layer_list, layer_locations, data_source_file, translations_folder, activations_folder, delete_existing, languages, activation_function, filerepo, layer_sparsity_type, sparsity)

    
    # translated language files
    for lang in languages:
        new_lang_source_file = data_source_file.replace('.csv', '-' + lang + ".csv")
        calculate_activations(thisProcess, log_file, s_autoencoder, activations_layer_list, layer_locations, new_lang_source_file, translations_folder, activations_folder, delete_existing, languages, activation_function, filerepo, layer_sparsity_type, sparsity)


# ============================= averages =============================
if process_averages:

    thisProcess = "averages"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    # ============================= delete/create averages folder =============================
    if delete_existing:
        if os.path.exists(averages_folder):
            shutil.rmtree(averages_folder)

    if not os.path.exists(averages_folder):
        os.makedirs(averages_folder)

    # for gpt2 the activation function is set to be an averageover feature activations
    if s_autoencoder == 'openai/gpt2':
        activation_function = ''
    
    calculate_averages(thisProcess, log_file, averages_folder, activations_folder, concept_pos_in_file, languages, activation_function, average_function)


# ============================= differences =============================
if process_similarities:

    thisProcess = "similarities"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)


    # ============================= create differences folder =============================
    if delete_existing:
        if os.path.exists(similarities_folder):
            shutil.rmtree(similarities_folder)

    if not os.path.exists(similarities_folder):
        os.makedirs(similarities_folder)

    df = pd.read_csv(ground_truth_file, encoding="utf-8")
    gt = pd.to_numeric(df["value"], errors="coerce").to_numpy()
    mask = np.isfinite(gt)
    gt = gt[mask]

    binary_target = False
    if is_binary_array(gt, tol=0.0):
        binary_target = True
        msg = "Ground truth variable is binary. Using the cartesian method of difference calculation."
        update_log(log_file, msg)

    else:
        msg = "Ground truth variable is continuous. Using the non-cartesian method of difference calculation."
        update_log(log_file, msg)
    
    calculate_similarities(thisProcess, log_file, ground_truth_file, languages, concept_pos_in_file, similarities_folder, averages_folder, activations_folder, data_start_file, binary_target, average_function)


# ============================= correlations =============================
if process_correlations:

    thisProcess = "correlations"
    # ============================= update log =============================
    msg = "starting process = " + thisProcess
    update_log(log_file, msg)

    calculate_correlations(thisProcess, log_file, similarities_folder, this_exp_folder, languages, average_function)


