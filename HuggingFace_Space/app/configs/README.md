# 🗂️ Configuration File Documentation

This document describes the parameters available in the [`/configs/config.json`](./config.json) file.

## Model

For the model architecture, such as the number of repeating model blocks.

- `in_dim`  
  - **Type**: `int`  
  - **Default**: `5`  
  - **Description**: Input dimension size.

- `intermediate_dim`  
  - **Type**: `int`  
  - **Default**: `128`  
  - **Description**: Size of intermediate hidden layers.

- `out_dim`  
  - **Type**: `int`  
  - **Default**: `1`  
  - **Description**: Output dimension size.

- `num_blocks`  
  - **Type**: `int`  
  - **Default**: `12`  
  - **Description**: Number of model blocks used.

- `dropout_rate`  
  - **Type**: `float`  
  - **Default**: `0.1`  
  - **Description**: Dropout rate for regularization.

## Logging

Used to configure the error/info logging behaviors.

- `log_to_file`  
  - **Type**: `bool`  
  - **Default**: `True`  
  - **Description**: Enable logging to a file.

- `log_file`  
  - **Type**: `str`  
  - **Default**: `logs/app.log`  
  - **Description**: Path to the log file.

- `logger_name`  
  - **Type**: `str`  
  - **Default**: `main`  
  - **Description**: Name of the logger instance.

- `log_level`  
  - **Type**: `str`  
  - **Default**: `INFO`  
  - **Description**: Logging level (e.g., INFO, DEBUG).

- `log_mode`  
  - **Type**: `str`  
  - **Default**: `w`  
  - **Description**: File mode for logging (e.g., 'w' for overwrite, 'a' for appending).

- `log_format`  
  - **Type**: `str`  
  - **Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`  
  - **Description**: Format of the logging method in the logger.

- `date_format`  
  - **Type**: `str`  
  - **Default**: `%Y-%m-%d %H:%M:%S`  
  - **Description**: Date and Time format of the log messages.

- `log_to_console`  
  - **Type**: `bool`  
  - **Default**: `True`  
  - **Description**: Enable logging to the console.