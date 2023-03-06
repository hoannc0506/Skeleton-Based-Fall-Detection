import tensorrt as trt
import logging
import argparse

class TRTEngineBuilder:
    def __init__(self, onnx_file_path, input_name, min_shapes, opt_shapes, max_shapes, fp16=False, int8=False, gpu_fallback=False, save_path='', verbose=True):
        trt.init_libnvinfer_plugins(None, "")
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        self.onnx_file_path = onnx_file_path
        self.input_name = input_name
        self.min_shapes = min_shapes
        self.opt_shapes = opt_shapes
        self.max_shapes = max_shapes
        self.fp16 = fp16
        self.int8 = int8
        self.gpu_fallback = gpu_fallback
        self.save_path = save_path
        self.verbose = verbose
        
    def build_engine(self):
        builder = trt.Builder(self.TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, self.TRT_LOGGER)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        
        with open(self.onnx_file_path, "rb") as model:
            parser.parse(model.read())
        # Define the optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape(self.input_name, self.min_shapes, self.opt_shapes, self.max_shapes)
        # Set config
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.int8:
            config.set_flag(trt.BuilderFlag.INT8)
        if self.gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
#         config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, (2 << 30) * 4)
        config.add_optimization_profile(profile)
        if self.verbose:
            logging.info("Building engine...")
            
        engine = builder.build_engine(network, config)
            
        if self.save_path is not None:
            if self.verbose:
                logging.info("Saving engine...")
            with open(self.save_path, "wb") as f:
                f.write(engine.serialize())
        return engine
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-path', type=str, default='pretrained/demo.onnx')
    parser.add_argument('--onnx-input', type=str, default='images')        
    parser.add_argument('--save-path', type=str, default='pretrained/saved.engine')    
    parser.add_argument('--min-bs', type=int, default=1)
    parser.add_argument('--max-bs', type=int, default=1)
    parser.add_argument('--optim-bs', type=int, default=1)
    parser.add_argument('--img-height', type=int, default=768)
    parser.add_argument('--img-width', type=int, default=960)

    
    args = parser.parse_args()
    onnx_path = args.onnx_path
    onnx_input = args.onnx_input
    save_path = args.save_path
    min_bs = args.min_bs
    max_bs = args.max_bs
    optim_bs = args.optim_bs
    h = args.img_height
    w = args.img_width
    
    trtbuilder = TRTEngineBuilder(
        onnx_file_path=onnx_path,
        input_name=onnx_input,
        min_shapes=(min_bs, 3, h, w),
        opt_shapes=(optim_bs, 3, h, w),
        max_shapes=(max_bs, 3, h, w),
        fp16=True,
        gpu_fallback=True,
        save_path=save_path,
        verbose=False
    )
    
    trtbuilder.build_engine()