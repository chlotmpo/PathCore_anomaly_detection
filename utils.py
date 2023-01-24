import torch
import torch
from torch import tensor
from torchvision import transforms
from sklearn import random_projection

def print_results(results : dict, method : str):
    
    print("\nFinal results of the computation")
    print(f"Average image rocauc: {results['average image rocauc']:.2f}")


def get_coreset_idx(
    z_lib : tensor, 
    n : int = 1000,
    eps : float = 0.90,
    float16 : bool = True,
    force_cpu : bool = False,
) -> tensor:
    """
        In order to obtain a reduced version of the Memory bank we use this function that will perform a greedy coreset supsampling. 
        The memory bnak become then fully searchable for larger image size and counts, allowing for patch-based comparison beneficial to anomaly detection. 
        With random subsampling, some significant information will be losed in the coverage of nominal features. 
        The coreset subsampling mechanism implemented in this method reduces the Memory bank in a better way and reduces inference time. 

        Conceptually, the selection of basic groups aims to find a subset so that the solutions of problems on A can be more closely 
        and above all more quickly approached by those calculated on S. 

        5 parameters characterized this function : 

        - tensor_list : This is a tensor representing a list of tensors that will be subsampled

        - n : Corresponds to the number of tensors that will be kept in the subsampling. It has been calculated based on a percentage given (the percentage of the Memory bank
        that we want to keep) multiplied by the number of tensors initially in the Memory bank

        - eps : This is a float that corresponds to the epsilon value that will be used in the random projection 

        - float16 : Boolean to determine if we want to use the 16-bit float precision or not

        - force_cpu: Boolean to determine if we want to use the CPU for the computations or not

        To better understand the following algorithm, first a random projection is performed on the tensor_list, then a list that will store indices is initialized
        The algorithm iterates from 0 to n -1 :
        - If this is the first iteration, the last_item and min_distances variables are initialized
        - For the rest of the iterations, for each one, the precedent variables are updates based on the distance between the rows of the tensor_list and last_item. 
          Then the row with the maximum value in min_distances is selected and it index is added to the list of indices. 
    """
    print("Beginning of the coreset subsampling reduction...")

    print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        # A random projection is performed on the tensor_list
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"   DONE.                 Transformed dim = {z_lib.shape}.")

    except ValueError:
        print( "   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx+1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    # Iteration from 0 to n-1, the number of tensors that will be kept in the subsampling
    for i in range(n-1):

        # The variables are updates based on the distances calculations
        # Broadcasting step
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) 
        # Iterative step
        min_distances = torch.minimum(distances, min_distances) 
        # Selection step
        select_idx = torch.argmax(min_distances) 

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    print("End of the coreset subsampling reduction")
    return torch.stack(coreset_idx)




