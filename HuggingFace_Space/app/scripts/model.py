# model.py
import torch
import torch.nn as nn

class ModuleLayer(torch.nn.Module):
    """Class for the individual layer blocks."""
    def __init__(self, intermediate_dim=32, dropout_rate=0.1):
        """Initializer for the 'ModuleLayer' class.
        Args:
            intermediate_dim (int): The dimension of the intermediate layer.
            dropout_rate (float): The dropout rate to apply after the ReLU activation.
        """
        super().__init__()
        self.mod_linear = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.mod_norm = torch.nn.LayerNorm(normalized_shape=intermediate_dim)
        self.mod_relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Forward pass of the layer block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor passing the input through the layer operations."""
        residual = x
        x = self.mod_linear(x)
        x = self.mod_norm(x)
        x = self.mod_relu(x)
        x = self.dropout(x)
        x += residual
        return x

class Agent(torch.nn.Module):
    """Class for Agent Structure using multiple Layer Blocks."""
    def __init__(self, cfg):
        """Initializer for the 'Agent' class.
        Args:
            cfg (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features=cfg["in_dim"], out_features=cfg["intermediate_dim"])
        self.layers = torch.nn.Sequential(*[ModuleLayer(intermediate_dim=cfg["intermediate_dim"], dropout_rate=cfg["dropout_rate"]) for _ in range(int(cfg["num_blocks"]))])
        self.out = torch.nn.Linear(in_features=cfg["intermediate_dim"], out_features=cfg["out_dim"])

    def forward(self, x):
        """Forward pass through the Agent's Layers.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            x (torch.Tensor): Output tensor after passing through the network.
        """
        x = self.linear(x)
        x = self.layers(x)
        x = self.out(x)
        return x
    
    def get_prediction(self, features):
        """Get the deterministic prediction on a single observation or a batch of observations.
        
        Parameters:
            features (torch.tensor): the agent's input features. Expected shape is either `(num_features,)` for a single observation
            or `(batch_size, num_features)` for a batch of observations.
        
        Returns:
            action (int or torch.tensor): 
                - If `features` is a single features (i.e., `features.dim() == 1`), returns a scalar `int` representing the chosen action. 

                - If `features` is a batch of features (i.e., `features.dim() > 1`),
                returns a `torch.Tensor` of `int`s, where each element is the
                chosen action for the corresponding observation in the batch"""
        # Ensure single samples have a batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0) # Add a batch dimension if it's a single batch of features
        with torch.no_grad():
            if not isinstance(features, torch.Tensor):  # Check if features is not already a tensor
                features = torch.tensor(features, dtype=torch.float)
            prediction = self.forward(features)  # Run a forward pass through the model
        if features.size(0) == 1:    # This method checks if there is only 1 element in a 1-D tensor
            return prediction.item() # Returns a Python scalar for a single observation
        else:
            return prediction # Returns a tensor of predictions