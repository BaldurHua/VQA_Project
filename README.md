# Visual Question Answering (VQA) Model with Stacked Attention Network (SAN)
## This repository contains a Visual Question Answering (VQA) model that utilizes Stacked Attention Networks (SAN) to process image and text inputs for answering questions related to an image.

### Project Overview
#### The model is trained on a VQA dataset, where it:

- Encodes images using a combination of ResNet and Inception blocks.
- Encodes questions using LSTM-based embeddings.
- Uses Stacked Attention Networks (SAN) to refine features from both modalities.
- Outputs an answer from a vocabulary of predefined answers.