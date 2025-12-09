import time
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class InferenceSession:
    def predict(self, inputs):
        raise NotImplementedError

    def benchmark(self, inputs, num_warmup=10, num_runs=100):
        # Warmup
        for _ in range(num_warmup):
            self.predict(inputs)
            
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.predict(inputs)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / num_runs
        return avg_latency

class PyTorchSession(InferenceSession):
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, inputs):
        with torch.no_grad():
            if isinstance(inputs, (list, tuple)):
                # Handle list of tensors (e.g. for DETR)
                inputs = [inp.to(self.device) for inp in inputs]
                return self.model(inputs)
            elif isinstance(inputs, dict):
                 # Handle dict inputs
                 inputs = {k: v.to(self.device) for k, v in inputs.items()}
                 return self.model(inputs)
            else:
                # Standard tensor
                inputs = inputs.to(self.device)
                return self.model(inputs)

class OnnxSession(InferenceSession):
    def __init__(self, onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
        except ImportError:
            raise ImportError("onnxruntime is not installed.")

    def predict(self, inputs):
        # Convert inputs to numpy
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
            ort_inputs = {self.input_names[0]: inputs}
        elif isinstance(inputs, (list, tuple)):
             # Assuming single input for ONNX usually, or map by index
             # If multiple inputs, we need to know mapping. 
             # For now, simplistic assumption: first input matches first arg
             ort_inputs = {self.input_names[0]: inputs[0].cpu().numpy()}
        else:
            raise ValueError("Unsupported input type for ONNX session")
            
        return self.session.run(None, ort_inputs)

class TensorRTSession(InferenceSession):
    def __init__(self, engine_path):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.cuda = cuda
            self.trt = trt
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
                
            self.context = self.engine.create_execution_context()
            
            # Allocate buffers
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.stream = cuda.Stream()
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
                    
        except ImportError:
            raise ImportError("tensorrt or pycuda is not installed.")

    def predict(self, inputs):
        # Copy input data to host buffer
        # Simplistic assumption: single input
        if isinstance(inputs, torch.Tensor):
            np.copyto(self.inputs[0]['host'], inputs.cpu().numpy().ravel())
        else:
             # Handle list/tuple
             np.copyto(self.inputs[0]['host'], inputs[0].cpu().numpy().ravel())

        # Transfer input data to the GPU.
        for inp in self.inputs:
            self.cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from the GPU.
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        # Synchronize the stream
        self.stream.synchronize()
        
        # Return outputs
        return [out['host'] for out in self.outputs]
