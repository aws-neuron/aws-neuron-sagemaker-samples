# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
# set the envvar before importing torch_neuronx
os.environ['NEURON_RT_NUM_CORES'] = '2'
import io
import time
import json
import torch
import numpy as np
import torch_neuronx

from PIL import Image
from wrapper import NeuronTextEncoder, UNetWrap, NeuronUNet
from diffusers.models.cross_attention import CrossAttention
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1-base"
dtype = DTYPE_REPLACE

def model_fn(model_dir, context=None):
    global model_id, dtype
    print("Loading model parts...")
    t=time.time()

    text_encoder_filename = os.path.join(model_dir, 'text_encoder/model.pt')
    decoder_filename = os.path.join(model_dir, 'vae_decoder/model.pt')
    unet_filename = os.path.join(model_dir, 'unet/model.pt')
    post_quant_conv_filename = os.path.join(model_dir, 'vae_post_quant_conv/model.pt')

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), None, set_dynamic_batching=False)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

    print(f"Done. Elapsed time: {(time.time()-t)*1000}ms")
    return pipe

def input_fn(request_body, request_content_type, context=None):
    if request_content_type == 'application/json':
        req = json.loads(request_body)
        prompt = req.get('prompt')
        num_inference_steps = req.get('num_inference_steps', 50)
        guidance_scale = req.get('guidance_scale', 7.5)
        if prompt is None or type(prompt) != str or len(prompt) < 5:
            raise("Invalid prompt. It needs to be a string > 5")
        if type(num_inference_steps) != int:
            raise("Invalid num_inference_steps. Expected int. default = 50")
        if type(guidance_scale) != float:
            raise("Invalid guidance_scale. Expected float. default = 7.5")
        return prompt,num_inference_steps,guidance_scale
    else:
        raise Exception(f"Unsupported mime type: {request_content_type}. Supported: application/json")

def predict_fn(input_req, model, context=None):
    prompt,num_inference_steps,guidance_scale = input_req
    return model(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]    

def output_fn(image, accept, context=None):
    if accept!='image/jpeg':
        raise Exception(f'Invalid data type. Expected image/jpeg, got {accept}')

    buffer = io.BytesIO()
    image.save(buffer, 'jpeg', icc_profile=image.info.get('icc_profile'))
    buffer.seek(0)
    return buffer.read()
