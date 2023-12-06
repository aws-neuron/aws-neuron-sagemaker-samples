# AWS Neuron SageMaker Samples

This repository contains SageMaker Samples using `ml.inf2` and `ml.trn1` instances for machine learning (ML) inference and training workloads on the AWS ML accelerator chips [Inferentia](https://aws.amazon.com/machine-learning/inferentia/) and [Trainium](https://aws.amazon.com/machine-learning/trainium/).

If you have additional SageMaker samples that you would like to contribute to this repository, please submit a pull request following the repository's contribution [guidelines](CONTRIBUTING.md).

Samples are organized by use case (training, inference)  below:

## Training

| Name | Description | Instance Type |
| --- | --- | --- |
| [BERT Fine-tuning on SageMaker](training/trn1-bert-fine-tuning-on-sagemaker) | Sample training notebook using Hugging Face Trainer API with leveraging [Neuron Persistent Cache](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/neuron-caching.html) | Trn1 |

## Inference

| Name | Description | Instance Type |
| --- | --- | --- |
| [BERT Inference on SageMaker](inference/inf2-bert-on-sagemaker) | Sample inference notebook using Hugging Face BERT model | Inf2, Trn1, Trn1n |
| [Stable Diffusion Inference on SageMaker](inference/stable-diffusion/) | How to compile and run HF Stable Diffusion model on SageMaker | Inf2, Trn1, Trn1n |
| [CLIP Inference on SageMaker](inference/inf2-clip-on-sagemaker/) |Sample notebook to compile and deploy a pretrained CLIP model | Inf2, Trn1, Trn1n |

## Getting Help

If you encounter issues with any of the samples in this repository, please open an issue via the GitHub Issues feature.

## Contributing

Please refer to the [CONTRIBUTING](CONTRIBUTING.md) document for details on contributing additional samples to this repository.


## Release Notes

Please refer to the [Change Log](releasenotes.md).
