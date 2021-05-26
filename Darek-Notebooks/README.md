# HPA Models and Inference Strategies from Darek

I have done all the modeling and experimentation via Jupyter notebooks. I am posting both training and inference notebooks in this folder. I will start with raw versions of the notebooks and iteratively clean them up for better readability.

# Models

I used three models from HPA 2018 winning solutions:

| Model        | Image Size | Weights     | Num Epochs | Data                              | Folds | GPU Time                        |
|--------------|------------|-------------|------------|-----------------------------------|-------|---------------------------------|
| Densenet     | 768        | bestfitting | 3          | train + public excl. classes 0,16 | 5     | 5 hours on single RTX3090 GPU   |
| Densenet     | 1536       | bestfitting | 6          | train + public excl. classes 0,16 | 1     | 3.5 hours on single RTX3090 GPU |
| Inception v3 | 1024       | pudae       | 5          | train + public excl. classes 0,16 | 1     | 3 hours on single RTX3090 GPU   |

# Inference strategies

I used two inference strategies I developed for the purpose of this competition. I have not found these in the literature. 

## Gap-Mask Inference

We train a regular image-level model. At inference, we modify the model architecture so that it takes two inputs - image and cell-mask. The image goes through the network alone until the GAP layer. At that point, we do element wise multiplication of the image activations and cell mask. From that point, an image gets expanded into a batch of single-cell images, and we apply the model head to that entire batch. 

We had two variations of this architecture - one for global average and max concat pooling, one for attention pooling.

![Gap-Mask-1](https://pbs.twimg.com/media/E2SRV0qWQAM7kju?format=jpg&name=medium)
![Gap-Mask-2](https://pbs.twimg.com/media/E2SRK5IWUAMli02?format=jpg&name=medium)

## Gridify Inference

When we started with single cell tiles, we resized each cell to the same size, e.g. 128x128. The problem is that this changed cell resolutions, some
got shrunk and some expanded. Inspired by (this paper)[https://arxiv.org/pdf/1906.06423.pdf], I looked for an approach that would line up the features model sees while training on full images, with features seen by the model when running inference on single cells.

The initial gridify approach was to take a single cell crop, copy-paste that crop with multiple augmentations into a 2048x2048 template, and then resize in the same way as during training. Finally, we converged on the following approach:

- Take 4x 512x512 crop from full image (independent of size) from each corner of the single cell
- Put those 4 crops into a single 1024x1024 image
- Resize to maintain same resolution as training. Eg. for the model trained on 768 size (from 2048), we resize into 384x384 cell tile

![Gridify](https://pbs.twimg.com/media/E1J0N0ZXEAYyffs?format=jpg&name=large)

# Validation

Due to potential leakage from similar images (same plate) across folds, we run all images through metric learning model (from bestfitting’s 2018 solution) and cluster similar images together with UMAP/DBSCAN. We treat the clusters as groups and divide data in folds based on group stratified multilabel approach. 

# Postprocessing

We reduce probabilities for cells based on negative signal, this gave us a small boost (~0.003): 
`preds[:,:18] = preds[:,:18] * (1 - preds[:,18]).unsqueeze(-1)`


