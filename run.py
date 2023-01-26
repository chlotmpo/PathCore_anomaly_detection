import warnings  
from data_loading import MVTecDataset, _CLASSNAMES, _CLASSNAMES_1, _CLASSNAMES_2, _CLASSNAMES_3, _CLASSNAMES_4
from patchcore_model import PatchCore
from utils import print_results

from typing import List
import torch
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

ALL_CLASSES = _CLASSNAMES
BATCH_CLASSES_1 = _CLASSNAMES_1
BATCH_CLASSES_2 = _CLASSNAMES_2 
BATCH_CLASSES_3 = _CLASSNAMES_3
BATCH_CLASSES_4 = _CLASSNAMES_4

def create_model(model_name: str):
    if model_name == "PatchCore":
        return PatchCore(f_coreset=.10, backbone_name="wideresnet50")
    else:
        raise ValueError(f"The model {model_name} do not exists or is not avaiable here")

def train_model(model, train_dataset):
    model.fit(train_dataset)

def evaluate_model(model, test_dataset, cls):
    return model.evaluate(test_dataset, cls = cls)

# Main method that will allow running the program
def run_model(model_name: str, classes: List):

    results = {}


    nb_datasets = len(classes)

    # This function iterates over the different datasets to train the model on it, test it with the test images and evaluate the performance
    for cls in classes:

        # Creation of the model
        model = create_model(model_name)
        print(f"\nThe model is runing on the {cls} dataset ...")

        # Load of the data related to the current dataset
        train_ds, test_ds = MVTecDataset(cls).get_dataloaders()

        # Train the model using the training data
        train_model(model, train_ds)

        # Use the test data to apply the trained model on it, try to classify it and evaluate the performance
        image_rocauc = evaluate_model(model, test_ds, cls)
        
        print("Here are the test results on the dataset", cls)
        print("ROC_AUC metric :", image_rocauc)

        results[cls] = [float(image_rocauc)]
        image_results = [v[0] for _, v in results.items()]

        # Compute the average score based on the scores obtained on the dataset tested
        average_image_roc_auc = sum(image_results)/len(image_results)


        total_results = {
            "per_class_results": results,
            "average image rocauc": average_image_roc_auc,
            "model parameters": model.get_parameters(),
        }

    return "After " + nb_datasets + " iterations performed on different datasets, here is the final average results obtained :\n " + total_results


run_model("PatchCore", ALL_CLASSES)

