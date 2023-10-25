import numpy as np
import pandas as pd
import os
from rdkit import Chem
from deepchem.molnet import load_bbbp, load_bace_classification
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec

from src.paths import *
from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from src.modelling.neural_net_model import NeuralNetModel
from src.modelling.neural_net import NeuralNet



def load_and_embed_cv_ss(dataset):

    if dataset == "bace":
        data = load_bace_classification(splitter="scaffold")
    else:
        data = load_bbbp(splitter="scaffold")

    df_train = data[1][0].to_dataframe()
    df_train = df_train[["ids", "y"]]
    df_train = df_train.rename(columns={"ids": "smiles", "y": "classification"})
    df_train["set"] = "train"

    df_valid = data[1][1].to_dataframe()
    df_valid = df_valid[["ids", "y"]]
    df_valid = df_valid.rename(columns={"ids": "smiles", "y": "classification"})
    df_valid["set"] = "valid"

    df_test = data[1][2].to_dataframe()
    df_test = df_test[["ids", "y"]]
    df_test = df_test.rename(columns={"ids": "smiles", "y": "classification"})
    df_test["set"] = "test"

    combined = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    df = combined.drop_duplicates()
    df = df.dropna()

    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=False))
    df["problems"] = df["mol"].apply(lambda x: len(Chem.DetectChemistryProblems(x)))
    df = df[df["problems"] == 0]
    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=True))

    sentence_df = pd.DataFrame()
    sentence_df["sentence"] = df.apply(lambda x: " ".join(MolSentence(mol2alt_sentence(x["mol"], 1)).sentence), axis=1)
    df["mol_sentence"] = df.apply(lambda x: MolSentence(mol2alt_sentence(x["mol"], 1)), axis=1)

    sentence_df["classification"] = df["classification"]
    sentence_df["set"] = df["set"]
    molsentence_df = df[["classification", "mol_sentence"]]

    sentence_df.to_csv(os.path.join(SMILES_DIRECTORY, "cleaned_" + dataset + "_ml.csv"))

    doc_list = sentence_df["sentence"].to_list()

    CV = CountVectorize(doc_list=doc_list, data_type="SMILES")

    tfidf = TFIDF(doc_list=doc_list, data_type="SMILES")
    tfidf.save_embedding(os.path.join(EMBEDDING_DIRECTORY, dataset + "_ml_tfidf.embeddding.npy"))

    CV.save_embedding(os.path.join(EMBEDDING_DIRECTORY, dataset + "_ml_cv.embeddding.npy"))


def load_and_embed_tfidf_ss(dataset):

    if dataset == "bace":
        data = load_bace_classification(splitter="scaffold")
    else:
        data = load_bbbp(splitter="scaffold")

    df_train = data[1][0].to_dataframe()
    df_train = df_train[["ids", "y"]]
    df_train = df_train.rename(columns={"ids": "smiles", "y": "classification"})
    df_train["set"] = "train"

    df_valid = data[1][1].to_dataframe()
    df_valid = df_valid[["ids", "y"]]
    df_valid = df_valid.rename(columns={"ids": "smiles", "y": "classification"})
    df_valid["set"] = "valid"

    df_test = data[1][2].to_dataframe()
    df_test = df_test[["ids", "y"]]
    df_test = df_test.rename(columns={"ids": "smiles", "y": "classification"})
    df_test["set"] = "test"

    combined = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    df = combined.drop_duplicates()
    df = df.dropna()

    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=False))
    df["problems"] = df["mol"].apply(lambda x: len(Chem.DetectChemistryProblems(x)))
    df = df[df["problems"] == 0]
    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=True))

    sentence_df = pd.DataFrame()
    sentence_df["sentence"] = df.apply(lambda x: " ".join(MolSentence(mol2alt_sentence(x["mol"], 1)).sentence), axis=1)
    df["mol_sentence"] = df.apply(lambda x: MolSentence(mol2alt_sentence(x["mol"], 1)), axis=1)

    sentence_df["classification"] = df["classification"]
    sentence_df["set"] = df["set"]
    molsentence_df = df[["classification", "mol_sentence"]]

    sentence_df.to_csv(os.path.join(SMILES_DIRECTORY, "cleaned_" + dataset + "_ml.csv"))

    doc_list = sentence_df["sentence"].to_list()

    tfidf = TFIDF(doc_list=doc_list, data_type="SMILES")

    tfidf.save_embedding(os.path.join(EMBEDDING_DIRECTORY, dataset + "_ml_tfidf.embeddding.npy"))

def get_labels_ss(label_df):
    train_labels = label_df[label_df["set"] == "train"]["classification"].to_numpy()
    test_labels = label_df[label_df["set"] == "test"]["classification"].to_numpy()
    valid_labels = label_df[label_df["set"] == "valid"]["classification"].to_numpy()

    return train_labels, test_labels, valid_labels

def get_embeddings_ss(embeddings, label_df):
    train_embeddings = embeddings[label_df["set"] == "train", :]
    test_embeddings = embeddings[label_df["set"] == "test", :]
    valid_embeddings = embeddings[label_df["set"] == "valid", :]

    return train_embeddings, test_embeddings, valid_embeddings


def train_nn_ss(model_name, train_embeddings, test_embeddings, valid_embeddings, train_labels, test_labels, valid_labels, sample):
    experiment = {
        "experiment": f"moleculenet_experiment",
        "batch_size": 200,
        "input_layer": train_embeddings.shape[1],
        "learning_rate": 0.001,
        "max_epochs": 50,
        "device": "cuda",
        "model_name": f"{model_name}",
        "shuffle_data": True,
        "hot_start_file_name": None,
        "log_metrics": False,
        "save_checkpoints": True
    }

    model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
    trainer = NeuralNet(model, experiment)
    trainer.main_train_test_on_data(train_data=train_embeddings, test_data=valid_embeddings,
                                                  train_labels=train_labels, test_labels=valid_labels)
    models = os.listdir(os.path.join(MODEL_DIRECTORY, experiment["experiment"] + "/" + experiment["model_name"]))
    paths = [os.path.join(os.path.join(MODEL_DIRECTORY, experiment["experiment"] + "/" + experiment["model_name"]), basename) for basename in models]
    best = max(paths, key=os.path.getctime)

    experiment["hot_start_file_name"] =  best
    model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
    tester = NeuralNet(model, experiment)
    test_metrics = tester.main_test_on_data(test_data=test_embeddings, test_labels=test_labels)

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]






