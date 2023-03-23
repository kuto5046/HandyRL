# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# neural nets

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F

from .util import map_r


def to_torch(x):
    return map_r(x, lambda x: torch.from_numpy(np.array(x)).contiguous() if x is not None else None)


def to_numpy(x):
    return map_r(x, lambda x: x.detach().numpy() if x is not None else None)


def to_gpu(data):
    return map_r(data, lambda x: x.cuda() if x is not None else None)


# model wrapper class

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def init_hidden(self, batch_size=None):
        if hasattr(self.model, 'init_hidden'):
            if batch_size is None:  # for inference
                hidden = self.model.init_hidden([])
                return map_r(hidden, lambda h: h.detach().numpy() if isinstance(h, torch.Tensor) else h)
            else:  # for training
                return self.model.init_hidden(batch_size)
        return None

    def forward(self, *args, **kwargs):
        """ 
        args[0]: state
        args[1]: prev_state
        """
        obses = args[0]
        state, prev_state = obses
        outputs = self.model.forward(state, prev_state, **kwargs)
        outputs['value'] = outputs['value'].tanh()
        return outputs 

    def inference(self, _x, hidden, **kwargs):
        # numpy array -> numpy array
        if hasattr(self.model, 'inference'):
            return self.model.inference(_x, hidden, **kwargs)

        self.eval()
        with torch.no_grad():
            # x, prev_x = _x 
            # xt = map_r(x, lambda x: torch.from_numpy(np.array(x)).contiguous().unsqueeze(0) if x is not None else None)
            # prev_xt = map_r(prev_x, lambda prev_x: torch.from_numpy(np.array(prev_x)).contiguous().unsqueeze(0) if prev_x is not None else None)
            # ht = map_r(hidden, lambda h: torch.from_numpy(np.array(h)).contiguous().unsqueeze(0) if h is not None else None)
            # outputs = self.forward(xt, prev_xt, ht, **kwargs)
            xt = map_r(_x, lambda x: torch.from_numpy(np.array(x)).contiguous().unsqueeze(0) if x is not None else None)
            ht = map_r(hidden, lambda h: torch.from_numpy(np.array(h)).contiguous().unsqueeze(0) if h is not None else None)
            outputs = self.forward(xt, ht, **kwargs)         
        return map_r(outputs, lambda o: o.detach().numpy().squeeze(0) if o is not None else None)

import io
import torch 
import onnx 
import onnxruntime as ort
def convert_onnx_model(model):
    model.eval()
    model.cpu()
    state = torch.zeros((1,19,48,48))
    prev_state = state.clone()
    args = (state, prev_state)
    # file_path = f"tmp.onnx"
    input_names = ["state", "prev_state"]
    output_names = ['robot_policy', 'value']
    # torch.onnx.export(model=model, args=args, f=file_path, input_names=input_names, output_names=output_names, opset_version=11)
    # onnx_model = ort.InferenceSession(file_path)
    onnx_io = io.BytesIO()
    torch.onnx.export(model=model, args=args, f=onnx_io, input_names=input_names, output_names=output_names, opset_version=11)
    onnx_io.seek(0)
    onnx_model = ort.InferenceSession(onnx_io.getvalue())
    return onnx_model 

# simple model
class OnnxModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = convert_onnx_model(model)

    def init_hidden(self, batch_size=None):
        if hasattr(self.model, 'init_hidden'):
            if batch_size is None:  # for inference
                hidden = self.model.init_hidden([])
                return map_r(hidden, lambda h: h.detach().numpy() if isinstance(h, torch.Tensor) else h)
            else:  # for training
                return self.model.init_hidden(batch_size)
        return None

    def forward(self, *args, **kwargs):
        """ 
        args[0]: state
        args[1]: prev_state
        """
        obses = args[0]
        state, prev_state = obses
        policy, value = self.model.run(None, {'state': state.numpy(), 'prev_state': prev_state.numpy()})
        return policy, value.tanh()

    def inference(self, _x, hidden, **kwargs):
        xt = map_r(_x, lambda x: torch.from_numpy(np.array(x)).contiguous().unsqueeze(0) if x is not None else None)
        ht = map_r(hidden, lambda h: torch.from_numpy(np.array(h)).contiguous().unsqueeze(0) if h is not None else None)    
        policy, value = self.forward(xt, ht, **kwargs)
        return {'robot_policy':policy.squeeze(0), 'value':value.squeeze(0)}
        # return map_r(outputs, lambda o: o.squeeze(0) if o is not None else None)



class RandomModel(nn.Module):
    def __init__(self, model, x):
        super().__init__()
        self.wrapped_model = ModelWrapper(model)
        self.hidden = self.wrapped_model.init_hidden()
        # outputs = self.wrapped_model.inference(x, self.hidden)
        # self.output_dict = {key: np.zeros_like(value) for key, value in outputs.items() if key != 'hidden'}

    def inference(self, *args, **kwargs):
        # return self.output_dict
        return self.wrapped_model.inference(args[0], self.hidden)
