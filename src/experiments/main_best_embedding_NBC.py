"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 14 September 2022
@Version  : 0.1
@Desc     :
"""

from src.paths import *
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utilities import embed_and_train, seed_script, create_metrics_df

seed_script()


# Initialize what processes should be run and initialize variables
MODEL_TYPE = "nbc"
DATA_LIST = ["cleaned_binary.csv", "cleaned_bace.csv", "cleaned_BBBP.csv"]
DATA_PATHS = [FASTA_DIRECTORY, SMILES_DIRECTORY, SMILES_DIRECTORY]
DATA_TYPES = ["FASTA", "SMILES", "SMILES"]
DATASET_NAMES = ["FASTA", "bace", "BBBP"]
NUM_SAMPLES = 50
TOPICS = [10, 100, 10]
DIMENSION = 100
N_JOBS = 7


experiment_folder = os.path.join(LOG_DIRECTORY, f"best_embedding_{MODEL_TYPE}")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

for i in range(len(DATA_LIST)):
    df = pd.read_csv(os.path.join(DATA_PATHS[i], DATA_LIST[i]))
    labels = df["classification"]

    if DATA_TYPES[i] == "FASTA":
        doc_list = df["sequence"].to_list()
    else:
        doc_list = df["sentence"].to_list()

    # CV
    metrics_list = Parallel(n_jobs=N_JOBS)(
        delayed(embed_and_train)(labels=labels,
                                 doc_list=doc_list,
                                 dataset=DATASET_NAMES[i],
                                 data_type=DATA_TYPES[i],
                                 embedding_type="cv",
                                 model_type=MODEL_TYPE,
                                 sample=sample) for sample in tqdm(range(NUM_SAMPLES)))
    experiment_folder = os.path.join(LOG_DIRECTORY, f"best_embedding_{MODEL_TYPE}")
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    test_metrics_file_name = os.path.join(experiment_folder,
                                          f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[i] + f"_cv_{MODEL_TYPE}.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[
        i] + f"_cv_{MODEL_TYPE}_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)

    # TFIDF
    metrics_list = Parallel(n_jobs=N_JOBS)(
        delayed(embed_and_train)(labels=labels,
                                 doc_list=doc_list,
                                 dataset=DATASET_NAMES[i],
                                 data_type=DATA_TYPES[i],
                                 embedding_type="tfidf",
                                 model_type=MODEL_TYPE,
                                 sample=sample) for sample in tqdm(range(NUM_SAMPLES)))
    experiment_folder = os.path.join(LOG_DIRECTORY, f"best_embedding_{MODEL_TYPE}")
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    test_metrics_file_name = os.path.join(experiment_folder, f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[
        i] + f"_tfidf_{MODEL_TYPE}.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[
        i] + f"_tfidf_{MODEL_TYPE}_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)

    # word2vec
    metrics_list = Parallel(n_jobs=N_JOBS)(
        delayed(embed_and_train)(labels=labels,
                                 doc_list=doc_list,
                                 dataset=DATASET_NAMES[i],
                                 data_type=DATA_TYPES[i],
                                 embedding_type="word2vec",
                                 model_type=MODEL_TYPE,
                                 sample=sample,
                                 dimension=DIMENSION) for sample in tqdm(range(NUM_SAMPLES)))
    experiment_folder = os.path.join(LOG_DIRECTORY, f"best_embedding_{MODEL_TYPE}")
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    test_metrics_file_name = os.path.join(experiment_folder, f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[
        i] + f"_word2vec_{MODEL_TYPE}.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder,
                                                f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[
                                                    i] + f"_word2vec_{MODEL_TYPE}_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)

    # LDA
    metrics_list = Parallel(n_jobs=N_JOBS)(
        delayed(embed_and_train)(labels=labels,
                                 doc_list=doc_list,
                                 dataset=DATASET_NAMES[i],
                                 data_type=DATA_TYPES[i],
                                 embedding_type="lda",
                                 model_type=MODEL_TYPE,
                                 sample=sample,
                                 topics=TOPICS[i]) for sample in tqdm(range(NUM_SAMPLES)))
    experiment_folder = os.path.join(LOG_DIRECTORY, f"best_embedding_{MODEL_TYPE}")
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    test_metrics_file_name = os.path.join(experiment_folder,
                                          f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[i] + f"_lda_{MODEL_TYPE}.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder,
                                                f"best_embedding_{MODEL_TYPE}_" + DATASET_NAMES[
                                                    i] + f"_lda_{MODEL_TYPE}_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)

