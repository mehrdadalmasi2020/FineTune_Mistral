# FineTune_Mistral
[![Downloads](https://static.pepy.tech/badge/FineTune-Mistral)](https://pepy.tech/project/FineTune-Mistral)

A Python package for fine-tuning the Mistral model and generating responses based on provided questions.
You can find instructions on how to prepare the training dataset from [FragenAntwortLLMGPU](https://pypi.org/project/FragenAntwortLLMGPU/) or [FragenAntwortLLMCPU](https://pypi.org/project/FragenAntwortLLMCPU/).

## Installation

```bash
pip uninstall FineTune_Mistral
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118       (refer https://pytorch.org/get-started/locally/)
pip install FineTune_Mistral
```

## Usage

``` python

# "In this library, the model used is mistralai/Mistral-7B-v0.1."

from FineTune_Mistral import FineTune_Mistral

dataset_path = 'path/to/your/dataset.jsonl'
cache_dir = 'path/to/cache'
output_dir = 'path/to/output'

fine_tuner = FineTune_Mistral(dataset_path, cache_dir)
fine_tuner.train_model(output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=2, per_device_eval_batch_size=1)

max_new_tokens = 500
temperature = 0.1
question = "What is the capital of France?"
response = fine_tuner.generate_response(question, max_new_tokens=max_new_tokens, temperature=temperature)
print("Generated response:", response)


```

### Explanation of Parameters

- **`dataset_path`**: Specifies the path to the dataset file in JSONL format. This dataset will be used for training the model.
- **`cache_dir`**: Directory for caching model files and other intermediate data. This helps in reusing downloaded files and speeding up the process.
- **`output_dir`**: Directory where the trained model, logs, and other outputs will be saved after training.
- **`num_train_epochs`**: The number of epochs to train the model. An epoch is one complete pass through the training dataset. More epochs can lead to better model performance but also increase training time.
- **`per_device_train_batch_size`**: The batch size used during training for each device (e.g., GPU). A larger batch size can speed up training but requires more memory.
- **`per_device_eval_batch_size`**: The batch size used during evaluation for each device. Similar to training batch size but used during model evaluation.
- **`question`**: The input question for which the model should generate a response.
- **`max_new_tokens`**: The maximum number of new tokens (words or pieces of words) the model should generate in response to the input question.
- **`temperature`**: The temperature rate for generating responses. Lower values make the output more deterministic and higher values make it more random.

# FineTune_Mistral

A Python package for fine-tuning the Mistral model and generating responses based on provided questions.

## Features

This package provides functionalities to fine-tune the Mistral model, a causal language model designed for generating coherent and contextually relevant text. 
The key features include:
- **Model Fine-Tuning**: Fine-tune the Mistral model on a custom dataset to adapt it to specific tasks or domains.
- **Response Generation**: Generate contextually relevant responses based on provided questions.
- **Easy Integration**: Simple and easy-to-use interface for integrating model fine-tuning and response generation into your applications.
- **Resource Management**: Efficiently manage computational resources with built-in cleanup functions.

The package specifically fine-tunes the "mistralai/Mistral-7B-v0.1" model, adapting it to the custom dataset provided by the user.

### Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Authors

- Mehrdad Almasi, Demival VASQUES FILHO, and Lars Wieneke

### Contact

For questions or feedback, please contact **Mehrdad.al.2023@gmail.com, demival.vasques@uni.lu, lars.wieneke@gmail.com**.