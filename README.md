# Comic Colorization using Pix2Pix

This project utilizes the Pix2Pix model, based on the paper *"Image-to-Image Translation with Conditional Adversarial Networks"*, to colorize comic images automatically.

## Project Overview
The aim of this project is to transform black-and-white comic images into fully colored versions using a deep learning model trained on a dataset of comic images. The Pix2Pix model, which leverages conditional GANs (Generative Adversarial Networks), is used for this image-to-image translation task.


## Dataset
The model is trained using the [Kaggle Comic Dataset](https://www.kaggle.com/datasets/a6226db5ae53d7dc4b0e7138843fff5d1e5f89f00c2655816e95a3057e048013). This dataset contains pairs of black-and-white comic images and their corresponding colored versions, which are used for training the Pix2Pix model.

## Setup
1. Clone this repository:

    ```bash
    git clone https://github.com/Soufiane-coder351/Comic-Coloring.git
    cd Comic-Coloring
    ```
2. Run the training script to start training the model:
    ```bash
    python3 train.py 
    ```
