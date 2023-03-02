import tensorrt as trt
import torch
import logging
import os
import glob
import cv2
import time
import torchvision
import argparse

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit




class TRTInferenceEngine:
    def __init__(self, engine_file_path, device='cpu', verbose=True):
        """
            TensorRT Inference Engine for Realtime Object Detection
        """
        trt.init_libnvinfer_plugins(None, "")
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        self.engine_file_path = engine_file_path
        self.device = device
        self.verbose = verbose
            
        self.engine = self._load_engine(self.engine_file_path)
        self.context = self.engine.create_execution_context()
        
        if self.verbose:
            self._verbose_binding_shape()
        
    def _verbose_binding_shape(self):
        for binding in self.engine:
            
            binding_idx = self.engine.get_binding_index(binding)
            binding_shape = self.engine.get_binding_shape(binding_idx)
            dtype = self.engine.get_binding_dtype(binding_idx)
            print('{}, shape: {}, type: {}'.format(binding, binding_shape, dtype))

        
    def _load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        logging.info("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def _overwrite_batch_size(self, shape: tuple, batch_size):
        assert len(shape) == 4
        
        typecasted_shape = list(shape)
        typecasted_shape[0] = batch_size
        
        return tuple(typecasted_shape) 
    
    def __call__(self, batched_images): # [batch_size, channels, height, width]
        
        batch = batched_images.shape[0]

        input_buffers, output_buffers = [], []
        outputs = []
        
        for binding in self.engine:

            binding_idx = self.engine.get_binding_index(binding)

            if self.engine.binding_is_input(binding):
                input_binding_shape = self.engine.get_binding_shape(binding_idx)

                self.context.set_binding_shape(binding_idx, self._overwrite_batch_size(input_binding_shape, batch))
                input_buffers.append(int(batched_images.data_ptr()))

            else:    
                output = torch.zeros(size=tuple(self.context.get_binding_shape(binding_idx)), dtype=torch.float32, device=self.device)
                outputs.append(output)
                output_buffers.append(int(output.data_ptr()))

        bindings = input_buffers + output_buffers
        
        self.context.execute_v2(bindings=bindings)
      
        return outputs




    
if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-path', type=str, default='pretrained/demo.engine')
    parser.add_argument('--batch-size', type=int, default=8)

    args = parser.parse_args()
    
    engine_path = args.engine_path
    batch_size = args.batch_size
    
    dataset_dir = '../dataset/Fall backwards'
    frame_paths = sorted(glob.glob(dataset_dir+'/*.png'))
    
    device = 'cuda:0'
    engine = TRTInferenceEngine(
        engine_path,
        device
    )
    
    from torch.utils.data import DataLoader
    inference_dataset = InferenceDataset(frame_paths)
    loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    
    t_total = 0
    skeletons = []

    for batch_inp, raw_imgs in loader:
        t_s = time.time()  
        batch_inp = batch_inp.to(device)
        output_batch = engine(batch_inp)[0]
        output_batch = non_max_suppression_kpt(output_batch, conf_thres=0.15, iou_thres=0.5, nc=1, nkpt=17)
       
        t_total += time.time() - t_s
        
   
    print(f'Process time: {t_total}')
    print(f'Num frame {len(inference_dataset)}')
    print(f'FPS: {len(inference_dataset)/t_total}')
    
    del engine
    
    
    