"""
Simple function to convert TensorDict to DataProto format.
"""

import numpy as np
import torch
from tensordict import TensorDict
from typing import Dict, Any
from verl import DataProto

def convert_tensordict_to_dataproto(tensordict_data: TensorDict, 
                                   non_tensor_data: Dict[str, Any] = None,
                                   meta_info: Dict[str, Any] = None) -> DataProto:
    """
    Convert TensorDict to DataProto format for reward function evaluation.
    
    Args:
        tensordict_data: TensorDict containing the batch data
        non_tensor_data: Optional dictionary containing non-tensor data (numpy arrays)
        meta_info: Optional dictionary containing metadata
        
    Returns:
        DataProto: The converted DataProto object
    """
    
    # Extract tensors from TensorDict
    tensors = {}
    for key, value in tensordict_data.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value
    
    # Handle non-tensor data
    if non_tensor_data is None:
        non_tensor_data = {}
    
    # Convert non-tensor data to numpy arrays with dtype=object
    processed_non_tensors = {}
    for key, value in non_tensor_data.items():
        if isinstance(value, np.ndarray):
            processed_non_tensors[key] = value
        else:
            # Convert to numpy array with object dtype
            try:
                processed_non_tensors[key] = np.array(value, dtype=object)
            except (TypeError, ValueError):
                continue
    
    # Create DataProto using the from_dict method
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=processed_non_tensors,
        meta_info=meta_info or {}
    ) 