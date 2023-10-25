"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : Paths for general items in the project.
"""
import os

LISTS_DIRECTORY = "data/lists"

DATA_DIRECTORY = "data"
SMILES_DIRECTORY = "data/smiles"

FASTA_DIRECTORY = "data/fasta"

MODEL_DIRECTORY = "models"
EMBEDDING_MODEL_DIRECTORY = os.path.join(MODEL_DIRECTORY, "embedding_models")
EXPERIMENT_DIRECTORY = "experiments"
LOG_DIRECTORY = os.path.join(MODEL_DIRECTORY, "logs")

EMBEDDING_DIRECTORY = os.path.join(DATA_DIRECTORY, "embeddings")

