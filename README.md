# MAX TEMPERATURE WEATHER PREDICTER
Attempted to create a trained model policy that can accurately predict the maximum temperature of a given day based on feature inputs.

## Data Source
The source is from [MaxPrestige/CA_Weather_Fire_Dataset_Cleaned](https://huggingface.co/datasets/MaxPrestige/CA_Weather_Fire_Dataset_Cleaned) and is a cleaned up version of [California Weather and Fire Prediction Dataset (1984–2025) with Engineered Features](https://zenodo.org/records/14712845). 


## Problems
Failed to get a low bias and low variance. 

The High Bias is due to the fact that the model was not fitting to the dataset and would not achieve a low Mean Absolute Loss on any of the Train or Validation Datasets. In general the Mean Absolute Error would be around 6, which is super high. 

This is in part due to the many outliers in the dataset after looking at the ranges of each cleaned column, specifically the Max Temp and Lagged Precipitation. 

The Model also has High variance as it achieves a high Mean Absolute Error on the Test Dataset (5.3667). 

## Possible Future Attemps
The Data needs to be modified in order for the model to fit the underlying patterns (e.g. removing entries that have a max temp of 85 or higher). Another method would be to gather more data where the max temperature is between 85 and 105. 

## Main Script Parameters:

You can customize the behavior of the training script by providing the following command-line arguments:

- --epochs

    - Type: int

    - Default: 8

    - Description: Number of training epochs to run.

- --learning_rate

    - Type: float

    - Default: 0.0003

    - Description: Learning rate used by the optimizer.

- --max_grad_norm

    - Type: float

    - Default: 3.0

    - Description: The Maximum L2 Norm of the gradients for Gradient Clipping.

- --dataloader_batch_size

    - Type: int

    - Default: 64

    - Description: Batch size used by the dataloaders for training, validation, and testing.

- --dataloader_pin_memory

    - Type: action='store_false' (boolean flag)

    - Default: True (if flag is not present)

    - Description: Disable pinned memory in dataloaders (enabled by default). Include this flag to disable it.

- --dataloader_num_workers

    - Type: int

    - Default: 0

    - Description: Number of subprocesses to use for data loading.

- --log_iterations

    - Type: int

    - Default: 2

    - Description: Frequency (in iterations) to log training progress.

- --eval_iterations

    - Type: int

    - Default: 2

    - Description: Frequency (in iterations) to evaluate the model.

- --use_cuda

    - Type: action='store_true' (boolean flag)

    - Default: False (if flag is not present)

    - Description: Enable CUDA for training if available. Include this flag to enable CUDA.

- --device

    - Type: str

    - Default: "cpu"

    - Description: Device to use for training (e.g., "cpu", "cuda:0"). This parameter overrides the --use_cuda flag if specified.

- --save_model

    - Type: action='store_true' (boolean flag)

    - Default: False (if flag is not present)

    - Description: Save the trained model after training. Include this flag to enable model saving.

- --model_output_path

    - Type: str

    - Default: "models/Spam-Classifier-GPT2-Model.pt"

    - Description: File path to save the trained model. Parent directories will be created if they do not exist.


### Example Commands:
- ```python Training_Loop_00_01.py```
    - Uses the default settings to run the script.
    
- ```pythonTraining_Loop_00_01.py --epochs=32  --log_iterations=4 --eval_iterations=8 --save_model```
    - Explanation: Run for 32 epochs and log the average batch iteration Mean Absolute Error Loss every 4 iterations. Evaluate the Policy under training every 8 epochs. Lastly, save the trained model. Uses the default save path. Uses default for everything else.
    
    
- ```pythonTraining_Loop_00_01.py --epochs=32  --use_cuda --save_model --model_output_path=models/first-trained-model.pt```
    - Explanation: Let the system detect if your system has GPU capabilities and has cuda available for training. During the training setup, the system dynamically sets the device variable for model training. Save the trained model using the specified location. Uses default for everything else.
