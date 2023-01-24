"""
This file contains the principal classes needed to perform the classification task
PatchCore represent the main class, as it is the main subject of this implementation, and is a subclass of the KNNExtractor class
The PatchCore class overrides the fit and predict methods of the KNNExtractor class to perform a specific type of image classification using a coreset sampling approach.
"""
import pdb
import torch
import torch.nn.functional as F
from torch import tensor
import numpy as np
import timm
import clip
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import models
from utils import get_coreset_idx
from neural_networks import ResNet50, WideResNet
from sklearn.metrics import roc_auc_score

class KNNExtractor(torch.nn.Module):
    def __init__(
        self,
        featExtract_model_name: str = "wideresnet50",
        output_indices: Tuple = (2,3),
        pool_last: bool = False,
        depth: int = 20,
        width_multiplier : float = 1.0,
        num_classes : int = 15, 
        dropout: float = 0.2,      
        featExtract_model = None
    ):
    
        self.depth = depth
        self.width_multiplier = width_multiplier
        self.num_classes = num_classes
        self.dropout = dropout
        self.output_indices = output_indices
        self.pool_last = pool_last
        self.backbone_name = featExtract_model_name
        self.preprocess = None
        """
            KNNExtractor is a class used as a superclass to the PatchCore. It is implemented to initialized a base for the PatchCore implementation
            It is based on the Pytorch library

            7 parameters characterized this class : 

            - feature_extractor_name : String variable that corresponds to the neural network model name that will be used as the feature extractor
            
            - output_indices : This Tuple represent the output indices to be used as the feature extractor
            
            - pool_last : Boolean that determines if an adaptive average pooling will be apply to the last feature extracted by the feature extractor or not

            - depth : Corresponds to the number of layers in each block of the WideResNet architecture 

            - width_multiplier : Parameters that will be used to controls the width of the layer, this value will be multiplied to the number of channels in each layer

            - num_classes : Number of output classes in the classification task

            - dropout : Dropout rate. It is used as a regularization technique to prevent overfitting

            After the base nn.Module class initialization, the __init__ method creates an instance of a neural network model as the feature extractor (here ResNet50),
            sets up the pooling layer if specified and the device to be used            
        """
        super().__init__()


        self.featExtract_model = timm.create_model(
			"wide_resnet50_2",
			out_indices=output_indices,
			features_only=True,
			pretrained=True,
		)

        # The following lines represents a try to create another pre-trained model to compare the performances
        # This one comes from CLIP image encoder and is an alternative of models pretrained on ImageNet
        # Unfortunately we were not able to achieve the computation correctly on time, but we leave the line of our tryings.

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.featExtract_model, self.preprocess = clip.load("RN50x64", device=device, jit=False)


        for param in self.featExtract_model.parameters():
            param.requires_grad = False

        # The feature extractor model is set to evaluation mode
        self.featExtract_model.eval()
        
        # If the variable pool_last is equal to True, an adaptative average pooling layer is applied
        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.featExtract_model_name = featExtract_model_name  
        self.output_indices = output_indices

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # The computation device is chosen and used 
        self.featExtract_model = self.featExtract_model.to(self.device)


    def __call__(self, tensor: torch.Tensor) -> Tuple:
        """
            The __call__ method in python, allows an object of a class to be called like a function
            This call method will allow to extract features from the input tensor using the neural network model predifined as the feature extractor
            It takes in parameters the tensors we want to extract features from.
            The output of this call method is a tuple composed of the extracted features
        """
        # The following disable the gradient tracking, they would not be stored in memory, there is no need to compute gradient in our case
        with torch.no_grad():

             # Extract the features from the input tensor using the feature extractor
            extracted_features = self.featExtract_model(tensor.to(self.device)) 

            # The following 2 lines corresponds to the implementation of the CLIP image encoder pre-trained model
            # extracted_features = self.featExtract_model.encode_image(tensor)
            # print(extracted_features[0].shape)
        
        # If the 'pool_last' attribute is set to True, 
        if self.pool:
            # It return a tuple containing all of the extracted feature maps except the last one, 
            # and the last feature map after being passed through the 'AdaptiveAvgPool2d' layer
            extracted_features = extracted_features[:-1], self.pool(extracted_features[-1]).to("cpu")
            
        # If the 'pool_last' attribute is set to False
        else:
            # It returns a tuple containing all of the extracted feature 
            extracted_features = [x.to("cpu") for x in extracted_features]
            
        return extracted_features
    
    def roc_auc(self,y_true, y_score):
        """
            This method is used to computed the roc metrics to evaluate the performance of the classification task
            ROC AUC means Receiver Operating Characteristics - Area Under the Curve 
            It is commonly used to evaluate the performance of a binary classification model. This metric is computed by plotting 
            the True Positive Rate (tpr) against the False Positive Rate (fpr) at various classification treshold. The area under the curve 
            is used as a summary of the model's performance. 
            A perfect classifier is represented by an AUC of 1
            This method return the value auc that corresponds at the area under the roc curve
        """
        # The first step is to create a list of pairs (y_true, y_score)
        data = list(zip(y_true, y_score))
        
        # Then the data have to be sorted based on the key value
        data = sorted(data, key=lambda x: x[1], reverse = True)
        # data.sort(key=lambda x: x[1])

        # Initialization of the variables representing the True Positive Rate (tpr) and the False Positive Rate (fpr)
        tpr = 0
        fpr = 0

        # Initialization of the auc variable that will store the value of the area under the curve
        auc = 0

        # for i in range(len(data)):
        for i, (y,s) in enumerate(data):

            # If the true label is 1 the true positive rate is incremented
            if y == 1:
                tpr += 1
            # If the true label is 0 the false positive rate is incremented
            else:   
                fpr += 1

            if i < len(data) - 1 and s != data[i + 1][1]:
                # The area under the curve is computed for this interval
                auc += (fpr / (fpr + tpr)) * (data[i + 1][1] - s)

        return auc
      
    def evaluate(self, test_data: DataLoader) -> Tuple[float, float]:
        """
            The evaluate method is used in this class to evaluate the model performance 
            The roc_auc method is called and used in this method
            The model performance is evaluate on the test data
            The output is the roc_auc scores for the image-level
        """
         # Initialization of the lists
        image_predictions = []
        image_labels = []

        # We iterate over the test data to evaluate the performance
        for sample, mask, label in test_data:
            
            # Retrieve the prediction for the current sample 
            z_score = self.predict(sample)
            
            # The z_score is added to the list of the images_predictions
            image_predictions.append(z_score.numpy())
            
            # And the label to the list images_labels
            image_labels.append(label[0].numpy())

        # Then we compute the roc_auc score for the images predictions, with the roc_auc method
        # image_roc_auc = self.roc_auc(image_labels, image_predictions)
        print("Sklearn roc_auc metric computation loading ...")
        image_roc_auc = roc_auc_score(image_labels, image_predictions)

        # Finally, the output is the roc_auc value as a tuple
        return image_roc_auc   
    
    def get_parameters(self, extra_params : dict = None) -> dict:
	    return {
			"backbone_name": self.backbone_name,
			"out_indices": self.output_indices,
			**extra_params,
            }



class PatchCore(KNNExtractor) :
    """
        This is the PatchCore class, which is a subclass of the KNNExtractor class
        This class is the main one and implement the main functionality of the anomaly detection 
        The __init__ method of this class override the one from the KNNExtractor class and have 3 more parameters :

         - f_coreset : This is a float that correspond to the percentage defined and to use for the coreset sampling method. 
         It represents the fraction of the number of training samples that we want to keep from the Memory bank

         - backbone_name : This string specify the name of the neural network to use as a backbone 

         - coreset_eps : This float corresponds to the sparse projection paramater and is used for selection a random subset of points as the coreset

         The PatchCore override the evaluate method of the KNNExtractor class and have two more functions : the fit() and the predict() that will 
         respectively fit the defined model to the training data ant predict the value on the test data using the trained model
    """ 
    def __init__(
        self,
        f_coreset: float = 0.01, 
        backbone_name: str = "wideresnet50", 
        coreset_eps: float = 0.90, 
        out_indices: Tuple = None, 
        pool_last: bool = False, 

    ): 
        # Initialize the parent class
        super().__init__(backbone_name, output_indices=(2,3), pool_last=pool_last)

        # Store the additional parameters
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.image_size = 224
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.n_reweight = 3
        self.memory_bank = []
        self.resize = None
        self.featExtract_model_name = backbone_name


    def fit(self, train_dl):
        """
            This method is used to fit the model using the training data. It will extract the features from the samples of the training data using the backbone network
            and then store the obtained patches in a patch memory. Then this memory will be subsampled using the coreset sampling method 
        """
        print("Beginning of the fit method...")
        # First, we initialize an empty list that will be used to store the patches created from the training data
        memory_bank = []

        # This variable will be used to store the size of the largest feature maps
        largest_fmap_size = None

        # The function iterate over the training data to train the model
        for sample in train_dl:

            # The features maps are extracte from the sample ( the self(sample) call the __call__ method from the KNNExtractor class)
            # image_input = self.preprocess(sample[0]).unsqueeze(0).to(self.device) # This line was a try to implement the CLIP image encoder pretrained model
            feature_maps = self(sample[0])


            # If the size of the largest feature map has not been set,
            # We have to set this size to the size of the current feature map
            if largest_fmap_size is None:
                largest_fmap_size = feature_maps[0].shape[-2:]

                # Using the Adaptative Average Pooling, the resize layer with the size of the largest feature map is initialized (TO VERIFY)
                self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)

            # The features extracted are resized using this computed value
            resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]

            # A patch is created, it is composed of several features maps
            patch = torch.cat(resized_maps, 1)

            # A reshape is needed, the patch is flattenned and then transposed
            patch = patch.reshape(patch.shape[1], -1).T

            # Finally, add the patch to the memory bank
            memory_bank.append(patch)

        # The patches in the memory bank are then concatenated in a single tensor
        self.memory_bank = torch.cat(memory_bank, 0)


        # If the chosen percentage to use as a coreset is less than 1
        # Then select a subset of the patch library as the coreset
        if self.f_coreset < 1:
            self.coreset_idx = get_coreset_idx(
                self.memory_bank,
                n=int(self.f_coreset * self.memory_bank.shape[0]),
                eps=self.coreset_eps,
            )
            self.memory_bank = self.memory_bank[self.coreset_idx]

       

    def predict(self, sample):
        """
            The predict method is applied to the test data to classify an image as a good one or as an outlier (detected as an anomaly)
            First, the patch is extracted from the sample and then a distance is computed between the patch and the patches in the memory bank
            The patch from the memory with the smallest distance is identified. 
            Then a reweighting is applied to this patch based on the distance between this patch and the nearest neighbors in the memory bank.
            The resulting weight is then returned.
        """
        # These first steps are the same that are applied on the training data when fitting the model
        feature_maps = self(sample)
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        # Compute distancew between every patch of the sample with every feature of the memory bank
        dist = torch.cdist(patch,self.memory_bank) 

        # Find for every path of the sample the closest patch in the memory bank, and the corresponding distance
        min_val, min_idx = torch.min(dist,dim=1) 

        # Among the list of the distances from the nearezs Memory Bank patch, we take the index of the biggest 
        s_idx = torch.argmax(min_val)  

        # Among the list of the distances from the neareast Memory Bank patch, we take the value of the biggest
        s_star = torch.max(min_val)  

        # Anomalous patch (the one with the biggest minimum distance)
        m_test = patch[s_idx].unsqueeze(0)  

        # Closest neighbours (in the memory bank)
        m_star = self.memory_bank[min_idx[s_idx]].unsqueeze(0)  

        # Find knn to m_star pt.1 | Computes the distances between the closest neighbour and all the other patches in the Memory Bank
        w_dist = torch.cdist(m_star,self.memory_bank) 

        # Pt.2 | Take the indexes of the top l neighbours in the Memory Bank 
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight,largest=False)  

        # Calculates the distance between the "worst" test patch and its knn patch in the Memory Bank
        m_star_knn = torch.linalg.norm(m_test - self.memory_bank[nn_idx[0, 1:]],dim=1)  

        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        # Apply the equation 7 from the paper
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # Return the anomaly sccore
        return s

    def get_parameters(self):
            return super().get_parameters({
                "f_coreset": self.f_coreset,
                "n_reweight": self.n_reweight,
            })