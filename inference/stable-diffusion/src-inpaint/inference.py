import os
os.environ['NEURON_RT_NUM_CORES'] = '2'
import torch
import torch.nn as nn
import torch_neuronx
import time
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention

import threading
import argparse
import sys
import copy
import PIL
import math
import json
import requests
import io
from io import BytesIO
import base64
from PIL import Image


model_id = "stabilityai/stable-diffusion-2-inpainting"
dtype = torch.float32

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]

# Optimized attention
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # Check for square matmuls
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled

def model_fn(model_dir, context=None):
    global model_id, dtype
    print("Loading model parts...")
    t=time.time()

    text_encoder_filename = os.path.join(model_dir, 'text_encoder/model.pt')
    decoder_filename = os.path.join(model_dir, 'vae_decoder/model.pt')
    unet_filename = os.path.join(model_dir, 'unet/model.pt')
    post_quant_conv_filename = os.path.join(model_dir, 'vae_post_quant_conv/model.pt')

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0,1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

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
        init_image = req.get('init_image')
        mask_image = req.get('mask_image')
        height = 512
        width = 512

        if prompt is None or type(prompt) != str or len(prompt) < 5:
            raise("Invalid prompt. It needs to be a string > 5")

        return prompt,init_image,mask_image,height,width
    else:
        raise Exception(f"Unsupported mime type: {request_content_type}. Supported: application/json")

def predict_fn(input_req, model, context=None):
    prompt,init_image,mask_image,height,width = input_req
    init_image_input = Image.open(io.BytesIO(base64.b64decode((init_image)))).convert("RGB").resize((width, height))
    mask_image_input = Image.open(io.BytesIO(base64.b64decode((mask_image)))).convert("RGB").resize((width, height))
    return model(prompt,image=init_image_input, mask_image=mask_image_input, height=height, width=width).images[0]    

def output_fn(image, accept, context=None):
    if accept!='image/jpeg':
        raise Exception(f'Invalid data type. Expected image/jpeg, got {accept}')

    buffer = io.BytesIO()
    image.save(buffer, 'jpeg', icc_profile=image.info.get('icc_profile'))
    buffer.seek(0)
    return buffer.read()
