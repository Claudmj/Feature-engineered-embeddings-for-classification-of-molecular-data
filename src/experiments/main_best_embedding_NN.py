import os
import pandas as pd
from tqdm import tqdm

from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from src.embeddings.word2vec import word2vec
from src.modelling.neural_net import NeuralNet
from src.modelling.neural_net_model import NeuralNetModel
from src.paths import *
from src.utilities import embed_and_train_nn, seed_script, create_metrics_df

seed_script()

# Initialize what processes should be run and initialize variables
DATA_LIST = ["cleaned_binary.csv", "cleaned_bace.csv", "cleaned_BBBP.csv"]
DATA_PATHS = [FASTA_DIRECTORY, SMILES_DIRECTORY, SMILES_DIRECTORY]
DATA_TYPES = ["FASTA", "SMILES", "SMILES"]
DATASET_NAMES = ["FASTA", "bace", "BBBP"]
NUM_SAMPLES = 50
DIMENSION = 200
TOPICS = [100, 100, 100]


experiment_folder = os.path.join(LOG_DIRECTORY, "best_embedding_nn")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

for i in tqdm(range(len(DATA_LIST))):
    df = pd.read_csv(os.path.join(DATA_PATHS[i], DATA_LIST[i]))
    labels = df["classification"].to_numpy()

    if DATA_TYPES[i] == "FASTA":
        doc_list = df["sequence"].to_list()
    else:
        doc_list = df["sentence"].to_list()


    metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                              index=range(NUM_SAMPLES))
    metrics_list = []
    for sample in tqdm(range(NUM_SAMPLES)):
        # CV
        results = embed_and_train_nn(labels=labels,
                                     doc_list=doc_list,
                                     dataset=DATASET_NAMES[i],
                                     data_type=DATA_TYPES[i],
                                     embedding_type="cv",
                                     sample=sample)
        metrics_list.append(results)
    test_metrics_file_name = os.path.join(experiment_folder, "best_embedding_nn_" + DATASET_NAMES[i] + "_cv_nn.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, "best_embedding_nn_" + DATASET_NAMES[i] + "_cv_nn_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)


    metrics_list = []
    for sample in tqdm(range(NUM_SAMPLES)):
        # tfidf
        results = embed_and_train_nn(labels=labels,
                                     doc_list=doc_list,
                                     dataset=DATASET_NAMES[i],
                                     data_type=DATA_TYPES[i],
                                     embedding_type="tfidf",
                                     sample=sample)
        metrics_list.append(results)
    test_metrics_file_name = os.path.join(experiment_folder, "best_embedding_nn_" + DATASET_NAMES[i] + "_tfidf_nn.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder,
                                                "best_embedding_nn_" + DATASET_NAMES[i] + "_tfidf_nn_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)


    metrics_list = []
    for sample in tqdm(range(NUM_SAMPLES)):
        # word2vec
        results = embed_and_train_nn(labels=labels,
                                     doc_list=doc_list,
                                     dataset=DATASET_NAMES[i],
                                     data_type=DATA_TYPES[i],
                                     embedding_type="word2vec",
                                     sample=sample,
                                     dimension=DIMENSION)
        metrics_list.append(results)
    test_metrics_file_name = os.path.join(experiment_folder, "best_embedding_nn_" + DATASET_NAMES[i] + "_word2vec_nn.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, "best_embedding_nn_" + DATASET_NAMES[i] + "_word2vec_nn_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)


    metrics_list = []
    for sample in tqdm(range(NUM_SAMPLES)):
        # lda
        results = embed_and_train_nn(labels=labels,
                                     doc_list=doc_list,
                                     dataset=DATASET_NAMES[i],
                                     data_type=DATA_TYPES[i],
                                     embedding_type="lda",
                                     sample=sample,
                                     topics=TOPICS[i])
        metrics_list.append(results)
    test_metrics_file_name = os.path.join(experiment_folder, "best_embedding_nn_" + DATASET_NAMES[i] + "_lda_nn.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, "best_embedding_nn_" + DATASET_NAMES[i] + "_lda_nn_stats.csv")
    create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)

