# TextSumWatermark
# Watermarking for Large Language Models

Welcome to the TextSumWatermark repository, dedicated to watermarking large language models. This project is inspired by the groundbreaking paper [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) with custom enhancements guided by Professor Yue Dong. 


### Here's an overview of the recent additions and improvements to our codebase:


- We have expanded our codebase to accommodate summarization models such as **T5** and **Flan**, in addition to just decoder models.
- We've introduced evaluation capabilities, allowing you to assess different models performance on datasets like **CNN_DailyMail** and **Xsum** seamlessly.
- We've integrated a Gradio demo with the latest models.
- To provide a deeper understanding of the watermarking approach, we've added a script to generate statistical result plots.

***
## Overview



[//]: # (![Demo Image]&#40;https://github.com/MahdiMohseni0033/TextSumWatermark/blob/main/images/Demo_image.png&#41;)

![model_explain1](.asset/Demo_image.PNG)

The core idea is to insert a subtle but detectable watermark into the generations of a language model. This is done by biasing the model's next token predictions during generation towards a small "greenlist" of candidate tokens determined by the prefix context. The watermark detector then analyzes text and checks if an abnormally high fraction of tokens fall within these expected greenlists, allowing it to distinguish watermarked text from human-written text.

## Contents

The repository contains the following key files:

- `watermark_processor.py`: Implements the `WatermarkLogitsProcessor` class which inserts the watermark during generation, and the `WatermarkDetector` class for analyzing text.

- `utils.py`: Utility functions for loading models, datasets, and running the interactive demo.

- `normalizers.py`: Text normalizers to defend against textual attacks against the watermark detector. 

- `config.yaml`: Configuration file for model and watermark parameters.

- `main.py`: Main script to run watermark insertion and detection on a dataset.

- `gradio_demo.py`: Script to launch an interactive web demo of watermarking using Gradio.

- `homoglyphs.py`: Helper code for managing homoglyph replacements.

- README: This file.

## Prerequisites and Installation

To run TextSumWatermark effectively, please ensure that you have the following prerequisites in place:

- Python Version: We recommend using Python version 3.9 or higher.
- You'll also need to install the necessary libraries. You can do this by running the following command:
```pip install -r requirements.txt```

These libraries include PyTorch and Transformers, which are essential for the operation of this codebase. Once you have the prerequisites set up, you're ready to dive into watermarking large language 

## Usage 

There are two main ways to use the code:

### 1. CLI usage

The `main.py` script allows watermarking and detection on an input dataset specified in the config file:

```python main.py```


This will load the model, dataset, and parameters from `config.yaml`, run generation with and without watermarks on the dataset, and perform detection on the outputs, saving results to `results.csv`.


### 2. Interactive demo

To launch a live demo:

```python gradio_demo.py```


This will launch a Gradio web interface allowing interactive prompt generation and watermark detection with parameters adjustable via the UI.


## Configuration

The key parameters controlling the watermarking scheme are:

- `gamma`: Fraction of tokens to assign to the greenlist at each step. Lower values make the watermark stronger. 
- `delta`: Amount of bias/boost applied to greenlist tokens during generation. Higher values make the watermark stronger.

The detector can be configured with:

- `z_threshold`: z-score cutoff to call a positive watermark detection  
- `normalizers`: Text normalizations to run during detection like lowercasing.

See the paper for detailed discussion of settings and their effect on generation quality and detection accuracy.


## Contact

If you have any questions or comments about this project, please don't hesitate to contact us:

Email: [mahdimohseni0333@gmail.com](mailto:mahdimohseni0333@gmail.com)