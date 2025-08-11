# ./Training_Loop_00_01.py
# ## VERSIONS
# - 00_02: 
#     - Cleanup Testing
# - 00_01: 
#     - Diagnosing Exploding Gradients
# - 00_00: 
#     - Initial Version

# ## Imports

# from importlib.metadata import version
import pandas as pd
# import seaborn as sn
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import Module # For type hinting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import argparse

# ## Data Preparation

# ### Custom Dataset

class WeatherDataset(Dataset):
    """Dataset class For the CA Weather Fire Dataset"""
    def __init__(self, csv_file="./Data/CA_Weather_Fire_Dataset_Cleaned.csv"):
        try:
            self.data = pd.read_csv(csv_file)   # Assign a pandas data frame
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {csv_file}")

        # Define feature and label columns
        self.feature_columns = self.data.columns.drop("MAX_TEMP")
        self.label_column = "MAX_TEMP"

    def __getitem__(self, index):
        features = self.data.loc[index, self.feature_columns].values
        
        label = self.data.loc[index, self.label_column] # Extract the label for the given index
        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )

    def __len__(self):
        return len(self.data)

# ### Data Pipeline

def data_pipeline(root_data_dir: str= "./Data", data_file_path: str="CA_Weather_Fire_Dataset_Cleaned.csv", data_splits_dir: str="DataSplits", batch_size: int=64, num_workers=0, pin_memory: bool=False, drop_last: bool=True) -> tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader, StandardScaler]:
    """This function prepares the train, test, and validation datasets.
    Args:
        root_data_dir (str): The root of the Data Directory
        data_file_path (str): The name of the original dataset (with .csv file extension).
        data_splits_dir (str): Path to the train, test, and validation datasets.
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
        scaler (StandardScaler): The scaler used to scale the features of the model input.
        """
    
    if not root_data_dir or not data_file_path or not data_splits_dir:  # Check for empty strings at the beginning
        raise ValueError("File and directory paths cannot be empty strings.")
    print(f"root_data_dir: {root_data_dir}")
    WEATHER_DATA_DIR = Path(root_data_dir)                  # Set the Data Root Directory

    WEATHER_DATA_CLEAN_PATH = WEATHER_DATA_DIR / data_file_path # Set the path to the complete dataset

    if WEATHER_DATA_CLEAN_PATH.exists():
        print(f"CSV file detected, reading from {WEATHER_DATA_CLEAN_PATH}")
        df = pd.read_csv(WEATHER_DATA_CLEAN_PATH)
    else:
        print(f"Downloading csv file from HuggingFace")
        try:
            df = pd.read_csv("hf://datasets/MaxPrestige/CA_Weather_Fire_Dataset_Cleaned/Data/CA_Weather_Fire_Dataset_Cleaned.csv")  # Download and read the data into a pandas dataframe
            os.makedirs(WEATHER_DATA_DIR, exist_ok=True)        # Create the Data Root Directory
            df.to_csv(WEATHER_DATA_CLEAN_PATH, index=False)     # Save the file, omitting saving the index
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during data download or saving: {e}")
    
    DATA_SPLITS_DIR = WEATHER_DATA_DIR / data_splits_dir
    TRAIN_DATA_PATH = DATA_SPLITS_DIR / "train.csv"
    TEST_DATA_PATH = DATA_SPLITS_DIR / "test.csv"
    VALIDATION_DATA_PATH = DATA_SPLITS_DIR / "val.csv"
    SCALER_PATH = DATA_SPLITS_DIR / "scaler.joblib"

    features = ['DAY_OF_YEAR', 'PRECIPITATION', 'LAGGED_PRECIPITATION', 'AVG_WIND_SPEED', 'MIN_TEMP']
    # features = ['PRECIPITATION','AVG_WIND_SPEED', 'MIN_TEMP']
    
    target = 'MAX_TEMP'

    if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH) and os.path.exists(VALIDATION_DATA_PATH) :
        print(f"Train, Test, and Validation csv datasets detected in '{DATA_SPLITS_DIR}', skipping generation")
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"Datasets not found in '{DATA_SPLITS_DIR}' or incomplete. Generating datasets...")
        os.makedirs(DATA_SPLITS_DIR, exist_ok=True)     # Create the Data Splits Parent Directory
        features = ['DAY_OF_YEAR', 'PRECIPITATION', 'LAGGED_PRECIPITATION', 'AVG_WIND_SPEED', 'MIN_TEMP']
        X = df[features]
        y = df[target]

        # split your data before scaling, shuffling the data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the training data ONLY. Need to use the scaler on all inputs that the model receives.
        # This means the mean and standard deviation are calculated from the training set.
        scaler.fit(X_train)

        # Transform the training, validation, and test data using the fitted scaler
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_validation_scaled = scaler.transform(X_validation)

        # Save the fitted scaler object
        try:
            joblib.dump(scaler, SCALER_PATH)
            print(f"Input scaler stored in: ({SCALER_PATH})")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred when saving Scaler: {e}")

        X_train_df = pd.DataFrame(X_train_scaled, columns=features)
        X_test_df = pd.DataFrame(X_test_scaled, columns=features)
        X_validation_df = pd.DataFrame(X_validation_scaled, columns=features)

        # Concatenate the features and labels back into a single DataFrame for each set
        train_data_frame = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
        test_data_frame = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)
        validation_data_frame = pd.concat([X_validation_df, y_validation.reset_index(drop=True)], axis=1)

        # Saving the split data to csv files
        train_data_frame.to_csv(TRAIN_DATA_PATH, index=False)
        test_data_frame.to_csv(TEST_DATA_PATH, index=False)
        validation_data_frame.to_csv(VALIDATION_DATA_PATH, index=False)

    print(f"Initializing DataLoaders and Returning")
    # Initialize the Different Datasets
    train_dataset = WeatherDataset(TRAIN_DATA_PATH)
    test_dataset = WeatherDataset(TEST_DATA_PATH)
    validation_dataset = WeatherDataset(VALIDATION_DATA_PATH)
    # Initialize the Different DataLoaders using the Datasets
    print(f"Creating DataLoaders with batch_size ({batch_size}), num_workers ({num_workers}), pin_memory ({pin_memory}). Training dataset drop_last: ({drop_last})")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    print(f"Training DataLoader has ({len(train_dataloader)}) batches, Test DataLoader has ({len(test_dataloader)}) batches, Validation DataLoader has ({len(validation_dataloader)}) batches")
    
    return (train_dataset, test_dataset, validation_dataset, train_dataloader, test_dataloader, validation_dataloader, scaler)
        

# ## Agent Architecture

# ### Layer Block

# %%
class LayerBlock(torch.nn.Module):
    """Class for the individual layer blocks."""
    def __init__(self, intermediate_dim=32, dropout_rate=0.1):
        super().__init__()
        self.Layer1 = torch.nn.Linear(in_features=intermediate_dim, out_features=intermediate_dim)
        self.Layer_Norm1 = torch.nn.LayerNorm(normalized_shape=intermediate_dim)
        self.ReLu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer_Norm1(x)
        x = self.ReLu(x)
        x = self.dropout(x)
        return x

# ### Weather Agent

class WeatherAgent(torch.nn.Module):
    """Class for Agent Structure using multiple Layer Blocks."""
    def __init__(self, cfg):
        super().__init__()
        self.L1 = torch.nn.Linear(in_features=cfg["in_dim"], out_features=cfg["intermediate_dim"])
        
        self.Layers = torch.nn.Sequential(
            *[LayerBlock(intermediate_dim=cfg["intermediate_dim"], dropout_rate=cfg["dropout_rate"]) for _ in range(cfg["num_blocks"])]
        )
        self.out = torch.nn.Linear(in_features=cfg["intermediate_dim"], out_features=cfg["out_dim"])

    def forward(self, x):
        x = self.L1(x)
        x = self.Layers(x)
        x = self.out(x)
        return x

# ## Main

# ### Log Iteration Functions

# %%
def log_iteration(batch_idx: int, total_batches: int, loss_value: float):
    """Logs the loss of the current batch."""
    print(f"Epoch batch [{batch_idx}/{total_batches}] | Loss: {loss_value:.7f}")

# %%
def log_epoch_iteration(epoch: int, avg_epoch_loss: float):
    """Log Current Metrics accumulated in the current epoch iteration.
    Args:
        epoch (int): the current iteration
        avg_epoch_loss (float): The average loss of the current epoch
    Returns:
        N/A
        """
    if avg_epoch_loss:
        print(f"=====================  [EPOCH ({epoch}) LOGGING]  =====================")
        print("| AVERAGES of THIS EPOCH:")
        print(f"| ACCUMULATED LOSS: {avg_epoch_loss:.7f}")
        print(f"===========================================================")
    
    else:
        print("No Data collected for this epoch to log")

# ### Evaluate Model Function

def evaluate_model(model: Module, dataloader: DataLoader, current_epoch: int = None, max_epochs: int=None, device: str = 'cpu') -> float:
    """
    Evaluates the model on a given dataset and returns the average loss.
    Args:
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
        print("Warning: Evaluation dataset is empty. Skipping evaluation.")
        return float('nan')
    
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.unsqueeze(dim=-1).to(device)
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)     # Calculate the average loss on the dataset

    if current_epoch and max_epochs:   # If the function was called in the training loop
        print(f"===================  [Epoch ({current_epoch}/{max_epochs})]  ===================")
        print(f"Entire Validation Dataset Average Loss: {avg_loss:.4f}")
        print(f"====================================================")

    else:   # If the function was called outside of the training loop
        print(f"===============================================")
        print(f"Entire Dataset Average Loss: {avg_loss:.4f} ")
        print(f"=====================================================")
            
    return avg_loss

# ### Train Model Function

def train_model(model_config: dict, train_dataloader: DataLoader, validation_dataloader: DataLoader, model: WeatherAgent = None, epochs=32, learning_rate=0.0003, max_grad_norm=0.5, log_iterations=10, eval_iterations=10, device="cpu") -> WeatherAgent:
    """The Model Training function.

    Args:
        model_config (dict): The base configurations for building the policies.
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
    print(f"Training Model on {device} with {epochs} main epochs, {learning_rate} learning rate, max_grad_norm={max_grad_norm}.")
    print(f"Logging every {log_iterations} epoch iterations, evaluating every {eval_iterations} epoch iterations.")

    agent = (model if model is not None else WeatherAgent(model_config)).to(device) # Create agent if nothing was passed, otherwise, create the agent. Send agent to device.

    optimizer = torch.optim.AdamW(params=agent.parameters(), lr=learning_rate, weight_decay=0.01) # Define the model optimization algorithm
    loss_fn = torch.nn.L1Loss(reduction='mean')     # Define the Loss function


    history = {'train_loss': [], 'val_loss': []}

    train_dataloader_length = len(train_dataloader)
    agent.train()   # Set agent to training mode
    for epoch in tqdm(range(epochs), desc=f">>>>>>>>>>>>>>>>>>>>>\nMain Epoch (Outer Loop)", leave=True):

        epoch_loss_total = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False)):           # Get a mini-batch of training examples from the dataloader
            # optimizer.zero_grad(set_to_none=True)       # Clear the gradients built up; Setting to None to improve performance
            optimizer.zero_grad()       # Clear the gradients built up; Setting to None to improve performance

            inputs, labels = inputs.to(device), labels.unsqueeze(dim=-1).to(device)   # Move the inputs and labels to the device

            agent_outputs = agent(inputs)       # Pass the inputs to the model and get the outputs.

            loss = loss_fn(agent_outputs, labels)      # Calculate the mini-batch loss
            epoch_loss_total += loss.item()
            
            loss.backward()         # Calculate the loss with respect to the model parameters
            torch.nn.utils.clip_grad_norm_(parameters=agent.parameters(), max_norm=max_grad_norm)   # Prevent the gradients from affecting the model parameters too much and reduce the risk of exploding gradients

            optimizer.step()      # Update the model's parameters using the learning rate

            # LOGGING LOSS OF CURRENT ITERATION
            if (batch_idx + 1) % log_iterations == 0:
                log_iteration(batch_idx=(batch_idx + 1), total_batches=train_dataloader_length, loss_value=loss.item())

        # CALCULATE AND STORE THE AVERAGE EPOCH LOSS
        epoch_avg_loss = epoch_loss_total / train_dataloader_length
        history["train_loss"].append(epoch_avg_loss)

        # LOG THE AVERAGE LOSS OF THE EPOCH
        log_epoch_iteration(epoch=epoch, avg_epoch_loss=epoch_avg_loss)

        # EVALUATE THE MODEL
        if (epoch + 1) % eval_iterations == 0:
            val_loss = evaluate_model(model=agent, dataloader=validation_dataloader, current_epoch=(epoch + 1), max_epochs=epochs, device=device)
            history["val_loss"].append(val_loss)
            agent.train()   # Set agent to training mode
        
    return agent.eval(), history


# ## Main

def main(args) -> int:
    print("SETTING UP FOR TRAINING")
    
    if args.device:     # Check if the user specified to use a CPU or GPU for training
        device = args.device
    else:
        if args.use_cuda:   # Check if the user wanted to use CUDA if available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAVE_LOCATION = "./models/Weather-Agent.pt"   # Define the model path and name of the trained model weights

    BASE_CONFIG={
    "in_dim": 5,
    "intermediate_dim": 128,
    "out_dim": 1,
    "num_blocks": 12,
    "dropout_rate": 0.1
}

    # --- Data Preparation Pipeline --- 
    try:
        (train_dataset, test_dataset, validation_dataset, train_dataloader, test_dataloader, validation_dataloader, scaler) = data_pipeline(batch_size=args.dataloader_batch_size)
    except ValueError as e:
        print(f"Caught an error: {e}")

    print("BEGINNING TRAINING SCRIPT")
    start_time=time.time()

    trained_policy, training_history = train_model(
        model_config=BASE_CONFIG,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        model=None,     # Create new model
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        log_iterations=args.log_iterations,
        eval_iterations=args.eval_iterations,
        device=device,
    )
    end_time=time.time()

    # --- Calculate Training Time --- 

    elapsed_time= end_time - start_time
    hrs = int(elapsed_time / 3600)
    min = int((elapsed_time % 3600) / 60)
    seconds_remaining = elapsed_time - (hrs * 3600 ) - (min * 60)

    print(f"FINISHED MODEL TRAINING. \nTRAINING TOOK: {hrs} Hours, {min} Minutes, and {seconds_remaining:.3f} Seconds")

    # --- Testing Trained Model --- 
    print("\nTESTING THE TRAINED POLICY:")
    test_loss = evaluate_model(model=trained_policy, dataloader=test_dataloader, current_epoch=None, max_epochs=None, device='cpu')

    # ---  Saving Model Section  ---   

    if args.save_model:     # Check if the user wants to save the trained model weights
        if args.model_output_path:     # Check if the user specified a target save location
            SAVE_LOCATION=args.model_output_path

        parent_dir = os.path.dirname(SAVE_LOCATION)

        # If parent_dir is empty, it means the SAVE_LOCATION is just a filename
        # in the current directory, so no new directories need to be created.
        if parent_dir and parent_dir != '.':
            try:
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Parent directory '{parent_dir}' created to store the model.")
            except OSError as e:
                print(f"Error creating directory {parent_dir}: {e}")
                SAVE_LOCATION='model.pt'      # Fall back to a default save location if problem occurs.
        
        try:
            torch.save(trained_policy.state_dict(), f=SAVE_LOCATION)
            print(f"Model weights saved in: {SAVE_LOCATION}")
        except Exception as e:
            print(f"Error saving model to {SAVE_LOCATION}: {e}")

    return 0

# Example usage (assuming you have a way to call this function, e.g., in a main block)
if __name__ == '__main__':
    # --- Begin Timing Main Script Execution Time ---
    main_start_time=time.time()

    parser = argparse.ArgumentParser(description="Train and evaluate a Regression Agent.")

    parser.add_argument('--epochs', type=int, default=8,
        help='(int, default=8) Number of training epochs to run.')

    parser.add_argument('--learning_rate', type=float, default=0.0003,
        help='(float, default=0.0003) Learning rate used by the optimizer.')
    
    parser.add_argument('--max_grad_norm', type=float, default=3.0,
        help='(float, default=3.0) The Maximum L2 Norm of the gradients for Gradient Clipping.')

    parser.add_argument('--dataloader_batch_size', type=int, default=64,
        help='(int, default=64) Batch size used by the dataloaders for training, validation, and testing.')

    parser.add_argument('--dataloader_pin_memory', action='store_false',
        help='(bool, default=True) Disable pinned memory in dataloaders (enabled by default).')

    parser.add_argument('--dataloader_num_workers', type=int, default=0,
        help='(int, default=0) Number of subprocesses to use for data loading.')

    parser.add_argument('--log_iterations', type=int, default=2,
        help='(int, default=2) Frequency (in iterations) to log training progress.')

    parser.add_argument('--eval_iterations', type=int, default=2,
        help='(int, default=2) Frequency (in iterations) to evaluate the model.')

    parser.add_argument('--use_cuda', action='store_true',
        help='(bool, default=False) Enable CUDA for training if available.')

    parser.add_argument('--device', type=str, default='cpu',
        help='(str, default="cpu") Device to use for training (e.g., "cpu", "cuda:0"). Overrides --use_cuda.')

    parser.add_argument('--save_model', action='store_true',
        help='(bool, default=False) Save the trained model after training.')

    parser.add_argument('--model_output_path', type=str, default='models/Weather-Agent.pt',
        help='(str, default="models/Weather-Agent.pt") File path to save the trained model.')


    args = parser.parse_args()

    ## End of ipynb testing

    ret = main(args)

    main_end_time=time.time()

    # --- Calculate Main Script Execution Time --- 

    elapsed_time= main_end_time - main_start_time
    hrs = int(elapsed_time / 3600)
    min = int((elapsed_time % 3600) / 60)
    seconds_remaining = elapsed_time - (hrs * 3600 ) - (min * 60)

    print(f"FINISHED MAIN SCRIPT\nOVERALL DURATION: {hrs} Hours, {min} Minutes, and {seconds_remaining:.3f} Seconds")
    if ret == 0:
        print("TERMINATING PROGRAM")
    else: 
        print("Main Scipt Error")
