# Compiling and Deploying HuggingFace Pretrained BERT on Inf2 on Amazon SageMaker

## Overview

In this tutotial we will compile and deploy a pretrained BERT base model from HuggingFace Transformers, using the [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers). 

The full list of HuggingFace’s pretrained BERT models can be found in the BERT section on this page https://huggingface.co/transformers/pretrained_models.html.

This Jupyter Notebook should run on a ml.c5.4xlarge SageMaker Notebook instance. You can set up your SageMaker Notebook instance by following the [Get Started with Amazon SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html) documentation. 


## Getting started

Run [inf2_bert_sagemaker.ipynb](./inf2_bert_sagemaker.ipynb) on SageMaker notebook instance.
The notebook was tested on a `ml.c5.4xlarge` SageMaker Notebook instance with `conda_pytorch_p310` kernel.
