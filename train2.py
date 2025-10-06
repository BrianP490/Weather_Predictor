"""This module contains the training loop for an Agent model."""

# # Creating Training Loop

# ## VERSIONS
# - 00_02:
#     - Created to try new Model architecture and align with project template
# - 00_01:
#     - Diagnosing Exploding Gradients
# - 00_00:
#     - Initial Version

# ## Imports

import json
import logging
import argparse
import time
import torch
import joblib
import os
import io
import sys
from importlib.metadata import version
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from argparse import Namespace
from logging import Logger # For type hinting
from torch.nn import Module # For type hinting
from typing import Dict, Any, List, Optional # For type hinting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from importlib.metadata import version

package_list = ['pandas', 'seaborn', 'matplotlib', 'torch', 'joblib', 'tqdm']
for package in package_list:
    try:
        print(f"{package} version: {version(package)}") # Raises PackageNotFoundError if not found
    except:
        print(f"❌ Package '{package}' not found. Please install it.")

# ## Global Variables

CONFIG_PATH = './configs/config.json'

# ## Data Pipeline

# ### Custom Dataset Class

class CustomDataset(Dataset):
    """Dataset class For the CA Weather Fire Dataset"""
    def __init__(self, csv_file="../Data/CA_Weather_Fire_Dataset_Cleaned.csv"):
        """Initializer for the Dataset class.
        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            label_column (str): The name of the column indicating the label."""
        try:
            self.data = pd.read_csv(csv_file)   # Assign a pandas data frame
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {csv_file}")

        # Define feature and label columns
        self.feature_columns = self.data.columns.drop("MAX_TEMP")
        self.label_column = "MAX_TEMP"

    def __getitem__(self, index):
        """Returns a tuple (features, label) for the given index.
        Args:
            index (int): Index of the data sample to retrieve.
        Returns:
            tuple: (features, label) where features is a tensor of input features and
            label is the corresponding label.
        """
        features = self.data.loc[index, self.feature_columns].values
        
        label = self.data.loc[index, self.label_column] # Extract the label for the given index
        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )

    def __len__(self):
        """Returns the amount of samples in the dataset."""
        return len(self.data)

# ### Utility Function

def clean_data(df: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    """Cleans the input DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
        logger (Logger): Logger object for logging information.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    show_dataframe_info = True  # Set to True to log DataFrame info

    # Log the initial state of the DataFrame
    if show_dataframe_info:
        logger.info(f"Initial DataFrame shape: {df.shape}")
        buffer = io.StringIO()  # Create a buffer to capture the info output
        df.info(buf=buffer) # Store the output into the buffer
        logger.info(f"Initial DataFrame info:\n " + buffer.getvalue())

    # Example cleaning steps (customize as needed)
    df = df.drop_duplicates() # Remove duplicates

    # Handle missing values (if any)
    if df.isnull().sum().sum() > 0:
        logger.info("Handling missing values...")
        df = df.dropna()  # Example: Drop rows with missing values
        logger.info(f"DataFrame shape after dropping missing values: {df.shape}")

    if show_dataframe_info:
        # Reinitialize the buffer to clear any previous content in order to log the 
        # final dataframe info
        buffer = io.StringIO()
        df.info(buf=buffer)
        logger.info(f"Final DataFrame info:\n " + buffer.getvalue())

    return df

# ### Data Pipeline Function

def data_pipeline(
        logger: Logger, dataset_url: str, root_data_dir: str= "../Data",
        data_file_path: str="Dataset.csv", data_splits_dir: str="DataSplits",
        scaler_dir = "Scalers", target_column: str="Target",
        extra_dropped_columns: Optional[List[str]]=None, batch_size: int=64,
        num_workers: int=0, pin_memory: bool=False, drop_last: bool=True) -> tuple[Dataset,
                                                                                   Dataset,
                                                                                   Dataset,
                                                                                   DataLoader,
                                                                                   DataLoader,
                                                                                   DataLoader,
                                                                                   StandardScaler]:
    """This function prepares the train, test, and validation datasets.
    Args:
        logger (Logger): The logger instance to log messages.
        dataset_url (str): The URL to download the dataset from, if not found locally.
        root_data_dir (str): The root of the Data Directory
        data_file_path (str): The name of the original dataset (with .csv file extension).
        data_splits_dir (str): Path to the train, test, and validation datasets.
        scaler_dir (str): Path to the feature and label scalers.
        target_column (str): The name of the target column to predict.
        extra_dropped_columns (List[str], optional): Additional columns to drop from the features.
        batch_size (int): The dataloader's batch_size.
        num_workers (int): The dataloader's number of workers.
        pin_memory (bool): The dataloader's pin memory option.
        drop_last (bool): The dataloader's drop_last option.

    Returns: 
        train_dataset (Dataset): Dataset Class for the training dataset.
        test_dataset (Dataset): Dataset Class for the test dataset.
        validation_dataset (Dataset): Dataset Class for the validation dataset.
        train_dataloader (DataLoader): The train dataloader.
        test_dataloader (DataLoader): The test dataloader.
        validation_dataloader (DataLoader): The validation dataloader.
        feature_scaler (MinMaxScaler): The scaler used to scale the features of the model input.
        """
    # Check for empty strings at the beginning
    if not root_data_dir or not data_file_path or not data_splits_dir:
        raise ValueError("File and directory paths cannot be empty strings.")
    logger.info(f"root_data_dir: {root_data_dir}")
    DATA_ROOT = Path(root_data_dir)

    DATA_CLEAN_PATH = DATA_ROOT / data_file_path # Set the path to the complete dataset

    if DATA_CLEAN_PATH.exists():
        logger.info(f"CSV file detected, reading from '{DATA_ROOT}'")
        df = pd.read_csv(DATA_CLEAN_PATH)
    else:
        logger.info(f"Downloading CSV file from Internet and saving into '{DATA_ROOT}'")
        try:
            os.makedirs(DATA_ROOT, exist_ok=True)       # Create the Data Root Directory
            df = pd.read_csv(dataset_url)  # Download and read the data into a pandas dataframe

            # Clean the data before saving
            df = clean_data(df, logger)

            df.to_csv(DATA_CLEAN_PATH, index=False)     # Save the file, omitting saving the row index
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred when downloading or saving the "
                               "dataset from '{dataset_url}' to '{DATA_CLEAN_PATH}'")
    
     # Define the paths for the data splits and scalers
    DATA_SPLITS_DIR = DATA_ROOT / data_splits_dir
    SCALER_DIR = DATA_ROOT / scaler_dir

    TRAIN_DATA_PATH = DATA_SPLITS_DIR / "train.csv"
    TEST_DATA_PATH = DATA_SPLITS_DIR / "test.csv"
    VALIDATION_DATA_PATH = DATA_SPLITS_DIR / "val.csv"

    FEATURE_SCALER_PATH = SCALER_DIR / "feature-scaler.joblib"
    # LABEL_SCALER_PATH = SCALER_DIR / "label-scaler.joblib"
    # Define the columns to drop from the features
    columns_to_drop = [target_column] + extra_dropped_columns

    if (os.path.exists(TRAIN_DATA_PATH) and 
        os.path.exists(TEST_DATA_PATH) and
        os.path.exists(VALIDATION_DATA_PATH)):
        print(f"Train, Test, and Validation csv datasets detected in '{DATA_SPLITS_DIR}', skipping generation and loading scaler(s)")
        try:
            feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred when loading scalers: {e}")    
    else:
        logger.info(f"Datasets not found in '{DATA_SPLITS_DIR}' or incomplete. Generating datasets...")
        # os.makedirs(MODEL_ROOT, exist_ok=True)
        os.makedirs(DATA_SPLITS_DIR, exist_ok=True)     # Create the Data Splits Parent Directory
        os.makedirs(SCALER_DIR, exist_ok=True)     # Create the Scaler Parent Directory

        # Initialize the StandardScaler
        feature_scaler = StandardScaler()
        # label_scaler = MinMaxScaler()

        try:
            df_features = df.drop(columns=columns_to_drop, inplace=False)
            df_labels = df[[target_column]]     # Instead of returning a pandas Series using "[]", return a dataframe using the "[[]]" to get a shape with (-1,1)
        except KeyError as e:
            raise KeyError(f"One or more specified columns to drop do not exist in the DataFrame: {e}")

        # split your data before scaling, shuffling the data
        X_train, X_inter, Y_train, Y_inter = train_test_split(df_features, df_labels, test_size=0.1, random_state=42)
        X_validation, X_test, Y_validation, Y_test = train_test_split(X_inter, Y_inter, test_size=0.5, random_state=42)


        # Fit the scaler on the training data ONLY. Need to use the scaler on all inputs that the model receives.
        # This means the mean and standard deviation are calculated from the training set.
        feature_scaler.fit(X_train)
        # label_scaler.fit(Y_train)

        # Transform the training, validation, and test data using the fitted scaler
        X_train_scaled = feature_scaler.transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        X_validation_scaled = feature_scaler.transform(X_validation)

        # Y_validation_scaled = label_scaler.transform(Y_validation)
        # X_test_scaled = feature_scaler.transform(X_test)
        # Y_test_scaled = label_scaler.transform(Y_test)

        # Save the fitted scaler object
        try:
            joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
            logger.info(f"Feature scaler stored in: ({FEATURE_SCALER_PATH})")
            # joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
            # logger.info(f"Feature scaler stored in: ({FEATURE_SCALER_PATH})")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred when saving Scaler(s): {e}")

        logger.info(f"Train Features Scaled Shape: {X_train_scaled.shape}")

        logger.info(f"Train Labels Shape: {Y_train.shape}")

        logger.info(f"Validation Features Scaled Shape: {X_validation_scaled.shape}")

        logger.info(f"Validation Labels Shape: {Y_validation.shape}")

        logger.info(f"Test Features Scaled Shape: {X_test_scaled.shape}")

        logger.info(f"Test Labels Shape: {Y_test.shape}")

        # Define the column names of the features and label
        features_names = df_features.columns
        # label_name = df_labels.columns

        # Scale the rest of the data; returns numpy arrays
        X_train_df = pd.DataFrame(X_train_scaled, columns=features_names)
        X_test_df = pd.DataFrame(X_test_scaled, columns=features_names)
        X_validation_df = pd.DataFrame(X_validation_scaled, columns=features_names)

        # Y_train_df = pd.DataFrame(Y_train_scaled, columns=label_name)
        # Y_test_df = pd.DataFrame(Y_test_scaled, columns=label_name)
        # Y_validation_df = pd.DataFrame(Y_validation_scaled, columns=label_name)

        # Concatenate the features and labels back into a single DataFrame for each set
        train_data_frame = pd.concat([X_train_df, Y_train.reset_index(drop=True)], axis=1)
        test_data_frame = pd.concat([X_test_df, Y_test.reset_index(drop=True)], axis=1)
        validation_data_frame = pd.concat([X_validation_df, Y_validation.reset_index(drop=True)], axis=1)

        # Saving the split data to csv files
        try:
            train_data_frame.to_csv(TRAIN_DATA_PATH, index=False)
            test_data_frame.to_csv(TEST_DATA_PATH, index=False)
            validation_data_frame.to_csv(VALIDATION_DATA_PATH, index=False)
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred when saving datasets to CSV files: {e}")

    # Creating Datasets from the stored datasets
    logger.info(f"INITIALIZING DATASETS")
    train_dataset = CustomDataset(TRAIN_DATA_PATH)
    test_dataset = CustomDataset(TEST_DATA_PATH)
    val_dataset = CustomDataset(VALIDATION_DATA_PATH)
    
    # Initialize the Different DataLoaders using the Datasets
    logger.info(f"Creating DataLoaders with 'batch_size'=({batch_size}), 'num_workers'=({num_workers}), 'pin_memory'=({pin_memory}). Training dataset 'drop_last'=({drop_last})")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, shuffle=True)
    validation_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, shuffle=False)

    print(f"Training DataLoader has ({len(train_dataloader)}) batches, Test DataLoader has ({len(test_dataloader)}) batches, Validation DataLoader has ({len(validation_dataloader)}) batches")
    
    return (train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, validation_dataloader, feature_scaler)
        

# ## Agent Architecture

# ### Module Layer

class ModuleLayer(torch.nn.Module):
    """Class for the individual layer blocks."""
    def __init__(self, intermediate_dim=32, dropout_rate=0.1):
        """Initializer for the 'ModuleLayer' class.
        Args:
            intermediate_dim (int): The dimension of the intermediate layer.
            dropout_rate (float): The dropout rate to apply after the ReLU activation.
        """
        super().__init__()
        self.mod_linear = torch.nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
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

# ### Agent Class

class WeatherAgent(torch.nn.Module):
    """Class for Agent Structure using multiple Layer Blocks."""
    def __init__(self, cfg):
        """Initializer for the 'Agent' class.
        Args:
            cfg (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features=cfg["in_dim"], out_features=cfg["intermediate_dim"])
        
        self.layers = torch.nn.Sequential(
            *[ModuleLayer(intermediate_dim=cfg["intermediate_dim"], dropout_rate=cfg["dropout_rate"]) for _ in range(cfg["num_blocks"])]
        )
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

# ## Training

# ### Log Iteration Functions

def log_iteration(logger: Logger, batch_idx: int, total_batches: int, loss_value: float) -> None:
    """Logs the loss of the current batch."""
    logger.info(f"Epoch batch [{batch_idx}/{total_batches}] | Loss: {loss_value:.7f}")

def log_epoch_iteration(logger: Logger, epoch: int, avg_epoch_loss: float) -> None:
    """Log Current Metrics accumulated in the current epoch iteration.
    Args:
        epoch (int): the current iteration
        avg_epoch_loss (float): The average loss of the current epoch."""
    if avg_epoch_loss:
        logger.info(f"=====================  [EPOCH ({epoch}) LOGGING]  =====================")
        logger.info("| AVERAGES of THIS EPOCH:")
        logger.info(f"| ACCUMULATED LOSS: {avg_epoch_loss:.7f}")
        logger.info(f"===========================================================")
    
    else:
        logger.warning("No Data collected for this epoch to log")

# ### Evaluate Model Function

def evaluate_model(logger: Logger, model: Module, dataloader: DataLoader, current_epoch: int = None, max_epochs: int=None, device: str = 'cpu') -> float:
    """
    Evaluates the model on a given dataset and returns the average loss.
    Args:
        logger (Logger): The logger instance to log messages.
        model (Module): The Model.
        dataloader (DataLoader): The dataloader to calculate average loss with.
        current_epoch (int): The current epoch [optional].
        max_epochs (int): The maximum number of epochs [optional].
        device (str): The device that the calculations will take place on.
    Returns:
        avg_loss (float): The calculated average loss.
    """
    model.eval()
    total_loss = 0.0
    # loss_fn = torch.nn.MELoss(reduction='sum') # Use reduction='sum' instead of 'mean' for total loss
    loss_fn = torch.nn.L1Loss(reduction='sum')
    if len(dataloader.dataset) == 0:
        logger.warning("Warning: Evaluation dataset is empty. Skipping evaluation.")
        return float('nan')
    
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.unsqueeze(dim=-1).to(device)
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)     # Calculate the average loss on the dataset

    if current_epoch and max_epochs:   # If the function was called in the training loop
        logger.info(f"===================  [Epoch ({current_epoch}/{max_epochs})]  ===================")
        logger.info(f"Entire Validation Dataset Average Loss: {avg_loss:.4f}")
        logger.info(f"====================================================")

    else:   # If the function was called outside of the training loop
        logger.info(f"===============================================")
        logger.info(f"Entire Dataset Average Loss: {avg_loss:.4f} ")
        logger.info(f"=====================================================")
            
    return avg_loss

# ### Training Function

def train_model(logger: Logger, model_config: dict, training_config: dict, train_dataloader: DataLoader, validation_dataloader: DataLoader, model: WeatherAgent = None, epochs=32, learning_rate=0.0003, max_grad_norm=0.5, log_iterations=10, eval_iterations=10, device="cpu") -> tuple[WeatherAgent, dict]:
    """The Model Training function.
    Args:
        logger (Logger): The logger instance to log messages.
        model_config (dict): The base configurations for building the policies.
        training_config (dict): The base configurations for training the model.
        train_dataloader (DataLoader): The dataloader for the training loop.
        validation_dataloader (DataLoader): The dataloader for the validation loop.
        model (WeatherAgent): The model to be trained.
        epochs (int): The number of times the outer loop is performed.
        learning_rate (float): The hyperparameter that affects how much the model's parameters learn on each update iteration.
        max_grad_norm (float): Used to promote numerical stability and prevent exploding gradients.
        log_iterations (int): Used to log information about the state of the Agent.
        eval_iterations (int): Used to run an evaluation of the Agent.
        device (str): The device that the model will be trained on.
    Returns: 
        agent (Module): The Trained Model in evaluation mode.
    """
    logger.info(f"Training Model on 'device'=({device}) with ({epochs}) main epoch(s), learning rate=({learning_rate}), max_grad_norm=({max_grad_norm}).")
    logger.info(f"Logging every ({log_iterations}) epoch iterations, and evaluating every ({eval_iterations}) epoch iterations.")

    agent = (model if model is not None else WeatherAgent(model_config)).to(device) # Create agent if nothing was passed, otherwise, create the agent. Send agent to device.

    optim_choice = training_config.get("optimizer", "AdamW").lower()
    if optim_choice == "adamw":
        optimizer = torch.optim.AdamW(params=agent.parameters(), lr=learning_rate, weight_decay=0.01)
    # Elif more optimizers are added in the future, they can be added here
    elif optim_choice == "adam":
        optimizer = torch.optim.AdamW(params=agent.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_choice}")
    
    loss_choice = training_config.get("loss_function", "mae").lower()
    if loss_choice == "mae":
        loss_fn = torch.nn.L1Loss(reduction='mean')      # Define the Loss function
    # Elif more loss functions are added in the future, they can be added here
    else:
        raise ValueError(f"Unsupported loss function: {loss_choice}")


    history = {'train_loss': [], 'val_loss': []}

    train_dataloader_length = len(train_dataloader)
    agent.train()   # Set agent to training mode

    # Loop over the number of epochs
    for epoch in tqdm(range(epochs), desc=f">>>>>>>>>>>>>>>>>>>>>\nMain Epoch (Outer Loop)", leave=True):

        epoch_loss_total = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False)):           # Get a mini-batch of training examples from the dataloader
            # optimizer.zero_grad(set_to_none=True)       # Clear the gradients built up; Setting to None to improve performance
            optimizer.zero_grad()       # Clear the gradients built up

            inputs, labels = inputs.to(device), labels.unsqueeze(dim=-1).to(device)   # Move the inputs and labels to the device

            agent_outputs = agent(inputs)       # Pass the inputs to the model and get the outputs.

            loss = loss_fn(agent_outputs, labels)      # Calculate the mini-batch loss
            epoch_loss_total += loss.item()
            
            loss.backward()         # Calculate the loss with respect to the model parameters
            torch.nn.utils.clip_grad_norm_(parameters=agent.parameters(), max_norm=max_grad_norm)   # Prevent the gradients from affecting the model parameters too much and reduce the risk of exploding gradients

            optimizer.step()      # Update the model's parameters using the learning rate

            # LOGGING LOSS OF CURRENT ITERATION
            if (batch_idx + 1) % log_iterations == 0:
                log_iteration(logger, batch_idx=(batch_idx + 1), total_batches=train_dataloader_length, loss_value=loss.item())

        # CALCULATE AND STORE THE AVERAGE EPOCH LOSS
        epoch_avg_loss = epoch_loss_total / train_dataloader_length
        history["train_loss"].append(epoch_avg_loss)

        # LOG THE AVERAGE LOSS OF THE EPOCH
        log_epoch_iteration(logger, epoch=epoch, avg_epoch_loss=epoch_avg_loss)

        # EVALUATE THE MODEL
        if (epoch + 1) % eval_iterations == 0:
            val_loss = evaluate_model(logger, model=agent, dataloader=validation_dataloader, current_epoch=(epoch + 1), max_epochs=epochs, device=device)
            history["val_loss"].append(val_loss)
            agent.train()   # Set agent to training mode

        
    return agent.eval(), history

# ## Miscelaneous Utility Functions

# ### Checks for current running mode

def is_notebook():
    """Checks if the code is running in a Jupyter notebook environment.
    Returns:
        bool: True if running in a Jupyter notebook, False otherwise."""
    try:
        shell = get_ipython().__class__.__name__
        
        # print(f"Detected shell: {shell}", flush=True)

        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other types
    except NameError:
        return False 

# ### Convert json to arg list

def json_to_arg_list(arg_dict) -> List[str]:
    """Converts a dictionary of arguments to a list suitable for argparse.
    Args:
        arg_dict (dict): Dictionary of arguments where keys are argument names and values are argument values.
    Returns:
        args (list): List of arguments formatted for argparse."""
    
    args = []
    for key, value in arg_dict.items():

        # print(f"Processing key: {key}, value: {value} type: {type(value)}")
        if value == False or value == True:
            if value:
                args.append(key)
        else:
            args.append(key)
            args.append(str(value))
    return args

# ### Creating Logger Function

def setup_logger(config: dict, propogate: bool=False) -> Logger:
    """Sets up and returns a named logger based on the provided config dictionary. The new logger will have different handlers based on the config.
    
    Args:
        config (dict): Dictionary containing logging configuration.
        propogate (bool): Whether to allow log messages to propagate to ancestor loggers.
    Returns:
        Logger: Configured logger instance."""

    logger_name = config.get('logger_name', 'main')
    log_to_file = config.get('log_to_file', True)   # Set whether to log to a logfile or not
    log_file = config.get('log_file', 'logs/app.log') # Get the log file path
    log_lvl = config.get('log_level', 'INFO')
    log_level = getattr(logging, log_lvl.upper(), logging.INFO) # Set fallback if invalid input
    log_mode = config.get('log_mode', 'w') # Set the log file mode
    log_format = config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = config.get('date_format', '%Y-%m-%d %H:%M:%S')
    log_to_console = config.get('log_to_console', True) # Set whether to log to console or not

    handlers = []   # Initialize the list of logging handlers

    logger = logging.getLogger(logger_name) # Create logger object with the specified name

    if not log_to_file and not log_to_console:
        # If no handlers are specified by the config
        print(f"Warning: No logging handlers configured for {logger_name}.\nVerbose Logging will be disabled.\nIn 'config/config.json', set ['log_to_file': true] or ['log_to_console': true] if you want to change the logging behavior.", flush=True)
    else:
        # Create log parent directory if it doesn't exist
        parent_dir = os.path.dirname(log_file) # Get the parent directory of the log file
        if parent_dir and parent_dir != '.':
            try:
                os.makedirs(name=parent_dir, exist_ok=True)
                print(f"Parent directory '{parent_dir}' used to store the log file.", flush=True)   # flush=True to ensure the message is printed immediately
            except OSError as e:
                print(f"Error creating directory '{parent_dir}': {e} INFO: Using default log file 'app.log' instead.", flush=True)
                log_file='app.log'      # Fall back to a default log file if problem occurs.
        
        # Remove all old handlers inherrited from the root logger
        for handler in logger.handlers[:]:  
            handler.close()
            logger.removeHandler(handler)
        
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)   # Create a formatter for the log messages

        if log_to_console:
            console_handler = logging.StreamHandler() # Initialize sending log messages to the console (stdout)
            console_handler.setFormatter(formatter)  # Set the formatter for the console handler
            handlers.append(console_handler)    # Add the console_handler to the list of handlers
        if log_to_file:
            file_handler = logging.FileHandler(filename=log_file, mode=log_mode)    # Initialize sending log messages to a file
            file_handler.setFormatter(formatter)  # Set the style for the console handler
            handlers.append(file_handler)   # Add the file_handler to the list of handlers

        # Add the handlers to the logger
        for handler in handlers:
            logger.addHandler(handler)

    logger.setLevel(log_level)  # Set logger minimum log level

    logger.propagate = propogate   # Prevent the log messages from being propagated to the root logger; gets rid of the root logger's default handlers, 
    
    return logger

# ### Retrieve Logger

def retrieve_logger(name: str = 'root') -> Logger:
    """
    Retrieves a named logger. If no handlers are attached, returns a root logger instance.
    Args:
        name (str): The name of the logger to retrieve.
    Returns:
        Logger: The logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        print(f"Retrieving root logger.", flush=True)
        return logging.getLogger()

    return logger

# ### Close file handlers and exit

def close_and_exit(logger: Logger, exit_code: int) -> None:
    """Closes all handlers of the logger and exits the program with the given exit code.
    Args:
        logger (Logger): The logger instance to close.
        exit_code (int): The exit code to terminate the program with.
    """
    print("Note: Closing any named loggers...", flush=True)
    used_handlers = logger.handlers[:]
    for handler in used_handlers:
        handler.close()
        logger.removeHandler(handler)

    print("Note: Closing any root logger...", flush=True)
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    
    print(f"Exiting program with exit code {exit_code}.", flush=True)
    
    if is_notebook():
        print("Detected Jupyter Notebook environment. Skipping sys.exit() to avoid kernel interruption.", flush=True)
    else:
        sys.exit(exit_code)

# ## Main Loop

def main(parser_args, global_config) -> int:
    """Main function to run the script pipeline.
    Args:
        parser_args: The arguments from the argument parser.
        global_config: The arguments from the config file.
    Returns:
        int: Exit code (0 for success, non-zero for failure).    """
    # Create logger object
    logger = retrieve_logger(global_config["logging"]["logger_name"])
    logger.info("STARTING MAIN FUNCTION")
    
    if parser_args.device == 'cpu' or parser_args.device == 'cuda':     # Check if the user specified to use a CPU or GPU for training
        DEVICE = parser_args.device
    else:
        if parser_args.use_cuda:   # Check if the user wanted to use CUDA if available.
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: 
            logger.info("Defaulted to using CPU for training.")
            DEVICE = 'cpu'

    SAVE_LOCATION = global_config["parser_defaults"]["model_output_path"]   # Get the model save destination path 

    BASE_CONFIG=global_config["model"]
    TRAINING_CONFIG=global_config["training"]
    PIPELINE_CONFIG=global_config["data"]
    
    # --- Data Preparation Pipeline --- 
    logger.info("RUNNING THE DATA PIPELINE")
    try:
        # Use dictionary unpacking to pass the PIPELINE_CONFIG parameters to the data_pipeline function
        (train_dataset, test_dataset, validation_dataset, train_dataloader, test_dataloader, validation_dataloader, feature_scaler) = data_pipeline(logger=logger, **PIPELINE_CONFIG, batch_size=parser_args.dataloader_batch_size, num_workers=parser_args.dataloader_num_workers, pin_memory=parser_args.dataloader_pin_memory, drop_last=True)
    
    except ValueError as e:
        logger.error(f"Caught a 'value' error: {e}", exc_info=True, stack_info=True)
        return 1
    except RuntimeError as e:
        logger.error(f"Caught a 'runtime' error: {e}", exc_info=True, stack_info=True)
        return 1

    logger.info("BEGINNING TRAINING SCRIPT")
    start_time=time.time()

    try:
        trained_policy, training_history = train_model(
            logger=logger, 
            model_config=BASE_CONFIG,
            training_config=TRAINING_CONFIG,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            model=None,     # Create new model
            epochs=parser_args.epochs,
            learning_rate=parser_args.learning_rate,
            max_grad_norm=parser_args.max_grad_norm,
            log_iterations=parser_args.log_iterations,
            eval_iterations=parser_args.eval_iterations,
            device=DEVICE,
        )
    except MemoryError as e:
        logger.error(f"Memory Error: {e}. Consider reducing the DataLoader's batch size or model complexity.", exc_info=True, stack_info=True)
        return 1
    except KeyboardInterrupt:
        logger.error("Training interrupted by user (KeyboardInterrupt).", exc_info=True, stack_info=True)
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}", exc_info=True, stack_info=True)
        return 1
    end_time=time.time()

    # --- Calculate Training Time --- 

    elapsed_time= end_time - start_time
    hrs = int(elapsed_time / 3600)
    min = int((elapsed_time % 3600) / 60)
    seconds_remaining = elapsed_time - (hrs * 3600 ) - (min * 60)

    logger.info(f"FINISHED MODEL TRAINING")
    logger.info(f"TRAINING TOOK: {hrs} Hours, {min} Minutes, and {seconds_remaining:.3f} Seconds")

    # --- Training History Section  ---
    logger.info("TRAINING HISTORY:")
    logger.info(training_history)

    # --- Testing Trained Model --- 
    logger.info("TESTING THE TRAINED POLICY:")
    test_loss = evaluate_model(logger=logger, model=trained_policy, dataloader=test_dataloader, current_epoch=None, max_epochs=None, device='cpu')

    # ---  Saving Model Section  ---   
    if parser_args.save_model:     # Check if the user wants to save the trained model weights
        logger.info("SAVING THE TRAINED POLICY:")
        if parser_args.model_output_path:     # Check if the user specified a target save location
            SAVE_LOCATION=parser_args.model_output_path

        parent_dir = os.path.dirname(SAVE_LOCATION) # Get the parent directory of the model's save location
        # Create the parent directory if it doesn't exist
        # If parent_dir is empty, it means the SAVE_LOCATION is just a filename or
        # in the current directory, so no new directories need to be created.
        if parent_dir and parent_dir != '.':
            try:
                os.makedirs(parent_dir, exist_ok=True)
                logger.info(f"Parent directory '{parent_dir}' used to store the model.")
            except OSError as e:
                logger.error(f"Error creating directory {parent_dir}: {e}", exc_info=True, stack_info=True)
                SAVE_LOCATION='model.pt'      # Fall back to a default save location if problem occurs.
        
        try:
            torch.save(trained_policy.state_dict(), f=SAVE_LOCATION)
            logger.info(f"Model weights saved in: {SAVE_LOCATION}")
        except Exception as e:
            logger.error(f"Error saving model to {SAVE_LOCATION}: {e}", exc_info=True, stack_info=True)

    return 0

# ## Main Process

# Call this function, during script execution; Main script entry point
if __name__ == '__main__':
    # --- Begin Timing Main Script Execution Time ---
    main_start_time=time.time()

    #  --- Load Config File ---
    try:
        with open(file=CONFIG_PATH, mode='r') as f:
            json_args = json.load(f)
    except FileNotFoundError:
        print(f"Config file not found. Please ensure '{CONFIG_PATH}' exists. Modify 'CONFIG_PATH' in Global Variables section if needed.", flush=True)
        exit(1)
    except Exception as e:
        print(f"Unexpected error loading config file: {e}. Exiting.")
        exit(1)

    # --- Logging Initialization Section ---
    log_to_console = json_args['logging']['log_to_console'] # Set whether to log to console or not
    log_to_file = json_args['logging']['log_to_file']   # Set whether to log to a logfile or not
    logger_config = json_args['logging']
    
    # Configure the root logger for any backup logging
    logging.basicConfig(level=logging.CRITICAL)  # Set root logger to highest level to suppress unwanted logs

    # Create the named logger only if the user wants to log to console or file
    if log_to_console or log_to_file:
        logger = setup_logger(config=logger_config, propogate=False)
    # Check if the user disabled both logging methods and resort to the root logger with no handlers
    elif not log_to_file and not log_to_console:
        print(f"========================================\nWarning: No logging handlers configured for logger, using root logger.\nVerbose Logging will be disabled.\nTraining progress will be shown using tqdm.\nIf you want to change the logging behavior:\nIn 'config/config.json', set ['log_to_file': true] or ['log_to_console': true]\n========================================\n ", flush=True)
        logger = logging.getLogger()  # Use the root logger
        
        # Remove all handlers inherrited from the root logger, if any exist
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    # --- Argument Parser Section ---
    parser = argparse.ArgumentParser(description="Train and evaluate a Regression Agent.")

    parser.add_argument('--epochs', type=int, default=json_args["parser_defaults"]["epochs"],
        help=f'(int, default={json_args["parser_defaults"]["epochs"]}) Number of training epochs to run.')

    parser.add_argument('--learning_rate', type=float, default=json_args["parser_defaults"]["learning_rate"],
        help=f'(float, default={json_args["parser_defaults"]["learning_rate"]}) Learning rate used by the optimizer.')
    
    parser.add_argument('--max_grad_norm', type=float, default=json_args["parser_defaults"]["max_grad_norm"],
        help=f'(float, default={json_args["parser_defaults"]["max_grad_norm"]}) The Maximum L2 Norm of the gradients for Gradient Clipping.')

    parser.add_argument('--dataloader_batch_size', type=int, default=json_args["parser_defaults"]["dataloader_batch_size"],
        help=f'(int, default={json_args["parser_defaults"]["dataloader_batch_size"]}) Batch size used by the dataloaders for training, validation, and testing.')

    parser.add_argument('--dataloader_pin_memory', action='store_true',
        help='(bool, default=False) Toggle pinned memory in dataloaders (disabled by default).')

    parser.add_argument('--dataloader_num_workers', type=int, default=json_args["parser_defaults"]["dataloader_num_workers"],
        help=f'(int, default={json_args["parser_defaults"]["dataloader_num_workers"]}) Number of subprocesses to use for data loading.')

    parser.add_argument('--log_iterations', type=int, default=json_args["parser_defaults"]["log_iterations"],
        help=f'(int, default={json_args["parser_defaults"]["log_iterations"]}) Frequency (in iterations) to log training progress.')

    parser.add_argument('--eval_iterations', type=int, default=json_args["parser_defaults"]["eval_iterations"],
        help=f'(int, default={json_args["parser_defaults"]["eval_iterations"]}) Frequency (in iterations) to evaluate the model.')

    parser.add_argument('--use_cuda', action='store_true',
        help='(bool, default=False) Enable CUDA for training if available.')

    parser.add_argument('--device', type=str, default=json_args["parser_defaults"]["device"],
        help=f'(str, default={json_args["parser_defaults"]["device"]}) Device to use for training (e.g., "cpu", "cuda:0"). Overrides --use_cuda.')

    parser.add_argument('--save_model', action='store_true',
        help='(bool, default=False) Save the trained model after training.')

    parser.add_argument('--model_output_path', type=str, default=json_args["parser_defaults"]["model_output_path"],
        help=f'(str, default={json_args["parser_defaults"]["model_output_path"]}) File path to save the trained model.')

    # --- Parse the argparse arguments ---

    if(is_notebook()):
        # --- Simulate command-line arguments for testing purposes (IPYNB TESTING ONLY) --- 
        json_sim_args = json_args["simulated_args"]
        # Convert JSON args to list for argparse
        simulated_args = json_to_arg_list(json_sim_args)

        parser_args = parser.parse_args(args=simulated_args)   # Overrides the default values with the simulated values
    else:   # Parse the argparse command-line arguments
        parser_args = parser.parse_args()
    
    # Log all of the passed parser arguments
    logger.info(parser_args)

    ## --- Call Main Script Section ---
    logger.info("CALLING MAIN SCRIPT...", exc_info=False, stack_info=False)

    # Call the main function with both the parser arguments and the 'config.json' file
    ret = main(parser_args, json_args)

    main_end_time=time.time()

    # --- Calculate Main Script Execution Time --- 

    elapsed_time= main_end_time - main_start_time
    hrs = int(elapsed_time / 3600)
    mins = int((elapsed_time % 3600) / 60)
    seconds_remaining = elapsed_time - (hrs * 3600) - (mins * 60)

    logger.info(f"FINISHED MAIN SCRIPT")
    logger.info(f"OVERALL DURATION: {hrs} Hours, {mins} Minutes, and {seconds_remaining:.3f} Seconds")

    # --- Determine final message based on return code ---
    if ret == 0:
        if not log_to_console and not log_to_file:
            print("FINISHED MAIN SCRIPT.\nCheck '/Models' folder for any saved model(s)", flush=True)
        else:
            logger.info("TERMINATING PROGRAM")
    else:
        if not log_to_console and not log_to_file:
            print("MAIN SCIPT ERROR", flush=True)
        else:
            logger.error("MAIN SCIPT ERROR")

    # --- Closes any logger file handlers and exits the program ---
    close_and_exit(logger, ret)


