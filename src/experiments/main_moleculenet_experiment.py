import pandas as pd
import os
from src.paths import *
from src.datasets.scaffold_split import load_and_embed_cv_ss, get_labels_ss, get_embeddings_ss, train_nn_ss, load_and_embed_tfidf_ss
from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from src.utilities import create_metrics_df, seed_script

seed_script()

from tqdm import tqdm

NUM_SAMPLES = 1


experiment_folder = os.path.join(LOG_DIRECTORY, "moleulenet_experiment")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)


load_and_embed_cv_ss(dataset="bace")

label_df = pd.read_csv(os.path.join(SMILES_DIRECTORY, "cleaned_bace_ml.csv"))
train_labels, test_labels, valid_labels = get_labels_ss(label_df)
embeddings = CountVectorize.load_embedding(os.path.join(EMBEDDING_DIRECTORY, "bace_ml_cv.embeddding.npy"))
train_embeddings, test_embeddings, valid_embeddings = get_embeddings_ss(embeddings, label_df)
model_name = "moleculenet_bace_cv_nn"

metrics_list = []
for sample in tqdm(range(NUM_SAMPLES)):
    results = train_nn_ss(model_name=model_name,
                          train_embeddings=train_embeddings,
                          test_embeddings=test_embeddings,
                          valid_embeddings=valid_embeddings,
                          train_labels=train_labels,
                          test_labels=test_labels,
                          valid_labels=valid_labels,
                          sample=sample)
    metrics_list.append(results)
test_metrics_file_name = os.path.join(experiment_folder, f"{model_name}.csv")
metrics_statistics_file_name = os.path.join(experiment_folder, f"{model_name}_stats.csv")
create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)




# BBBP

load_and_embed_tfidf_ss(dataset="BBBP")

label_df = pd.read_csv(os.path.join(SMILES_DIRECTORY, "cleaned_BBBP_ml.csv"))
train_labels, test_labels, valid_labels = get_labels_ss(label_df)
# embeddings = CountVectorize.load_embedding(os.path.join(EMBEDDING_DIRECTORY, "BBBP_ml_cv.embeddding.npy"))
embeddings = TFIDF.load_embedding(os.path.join(EMBEDDING_DIRECTORY, "BBBP_ml_tfidf.embeddding.npy"))
train_embeddings, test_embeddings, valid_embeddings = get_embeddings_ss(embeddings, label_df)
model_name = "moleculenet_BBBP_tfidf_nn"

metrics_list = []
for sample in tqdm(range(NUM_SAMPLES)):
    results = train_nn_ss(model_name=model_name,
                          train_embeddings=train_embeddings,
                          test_embeddings=test_embeddings,
                          valid_embeddings=valid_embeddings,
                          train_labels=train_labels,
                          test_labels=test_labels,
                          valid_labels=valid_labels,
                          sample=sample)
    metrics_list.append(results)
test_metrics_file_name = os.path.join(experiment_folder, f"{model_name}.csv")
metrics_statistics_file_name = os.path.join(experiment_folder, f"{model_name}_stats.csv")
create_metrics_df(NUM_SAMPLES, metrics_list, test_metrics_file_name, metrics_statistics_file_name)

