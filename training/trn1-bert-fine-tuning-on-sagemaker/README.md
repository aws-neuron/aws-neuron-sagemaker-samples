# Fine tuning BERT base model from HuggingFace on Amazon SageMaker

## Overview

This tutotial uses the Hugging Face transformers and datasets libraries with Amazon SageMaker to fine-tune a pre-trained BERT base model on binary text classification with using Hugging Face Trainer API.

A [pre-trained BERT base model](https://huggingface.co/bert-base-uncased) is available in the transformers library from Hugging Face. You’ll be fine-tuning this pre-trained model using the [Amazon Polarity dataset](https://huggingface.co/datasets/amazon_polarity) which classify the content into either positive or negative feedback.

You will see how [Neuron Persistent Cache](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/neuron-caching.html) can be used for SageMaker Training jobs. At the end of the training job, Neuron Persistent Cache is uploaded to S3 so the cache can be reused for the another training jobs.


## Getting started

Run [bert-base-uncased-amazon-polarity.ipynb](./bert-base-uncased-amazon-polarity.ipynb) either on SageMaker notebook instance or SageMaker Studio Notebook.

You can set up your SageMaker Notebook instance by following the [Get Started with Amazon SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html) or SageMaker Studio Notebook by following the [Use Amazon SageMaker Studio Notebooks](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks.html)

The notebook was tested on a `ml.t3.medium` SageMaker Studio Notebook with `Data Science` image.
