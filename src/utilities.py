"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : General useful functions
"""
import os
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from src.embeddings.word2vec import word2vec
from src.modelling.support_vector_machine_classifier import svm_master_class
from src.modelling.naive_bayes_classifier import nbc_master_class
from src.embeddings.LDA import LDA
from src.modelling.neural_net import NeuralNet
from src.modelling.neural_net_model import NeuralNetModel


seed_value = 0

def seed_script():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def read_list(file_name: str):
    with open(file_name, "r") as file:
        lines = file.read().split("\n")
        return lines

def save_list(file_name: str, list: list):
    with open(file_name, "w") as file:
        file.write("\n".join(list))


def embed_and_train(labels, doc_list, dataset, data_type, embedding_type, model_type, sample, topics=None, dimension=None):
    if embedding_type == "cv":
        cv = CountVectorize(doc_list=doc_list, data_type=data_type)
        embeddings = cv.embedding
    elif embedding_type == "tfidf":
        tfidf = TFIDF(doc_list=doc_list, data_type=data_type)
        embeddings = tfidf.embedding
    elif embedding_type == "word2vec":
        word2vec_model = word2vec(doc_list=doc_list, data_type=data_type, dimension=dimension)
        embeddings = word2vec_model.embedding
    elif embedding_type == "lda":
        lda_model = LDA(doc_list=doc_list, data_type=data_type, num_topics=topics)
        embeddings = lda_model.embedding

    train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2,
                                                                        random_state=seed_value)

    model_parameters = {
        "model_name": f"best_embedding_{model_type}_" + dataset + f"_cv_{model_type}",
        "dataset_name": dataset
    }

    if model_type == "svm":
        svm = svm_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels,
                               test_labels=test_labels, load_model=None)
        svm.train()
        svm.predict()
        svm.evaluate()
        test_metrics = svm.test_metrics
    elif model_type == "nbc":
        nbc = nbc_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels,
                               test_labels=test_labels, load_model=None, embedding_type="tfidf")
        nbc.train()
        nbc.predict()
        nbc.evaluate()
        test_metrics = nbc.test_metrics

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]


def create_metrics_df(num_samples, metrics_list, test_metrics_file_name, metrics_statistics_file_name):
    metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                              index=range(num_samples))
    metrics_df.iloc[:num_samples] = metrics_list
    metrics_df.to_csv(test_metrics_file_name)
    metrics_df = metrics_df.apply(pd.to_numeric)
    statistics_df = metrics_df.describe(include='all')
    statistics_df.to_csv(metrics_statistics_file_name)


def embed_and_train_nn(labels, doc_list, dataset, data_type, embedding_type, sample, topics=None, dimension=None):
    if embedding_type == "cv":
        cv = CountVectorize(doc_list=doc_list, data_type=data_type)
        embeddings = cv.embedding
    elif embedding_type == "tfidf":
        tfidf = TFIDF(doc_list=doc_list, data_type=data_type)
        embeddings = tfidf.embedding
    elif embedding_type == "word2vec":
        word2vec_model = word2vec(doc_list=doc_list, data_type=data_type, dimension=dimension)
        embeddings = word2vec_model.embedding
    elif embedding_type == "lda":
        lda_model = LDA(doc_list=doc_list, data_type=data_type, num_topics=topics)
        embeddings = lda_model.embedding

    experiment = {
        "experiment": "best_embedding_nn",
        "batch_size": 200,
        "input_layer": embeddings.shape[1],
        "learning_rate": 0.001,
        "max_epochs": 50,
        "device": "cuda",
        "model_name": "best_embedding_nn_" + dataset + f"_{embedding_type}_nn",
        "shuffle_data": True,
        "hot_start_file_name": None,
        "log_metrics": False,
        "save_checkpoints": False,
        "data": embeddings,
        "labels": labels
    }

    model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
    trainer = NeuralNet(model, experiment)
    test_metrics = trainer.main_train_test()

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]

