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
        elif isinstance(inputs, np.ndarray):
             ort_inputs = {self.input_names[0]: inputs}
        elif isinstance(inputs, dict):
             ort_inputs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
             # Detection dataloader often yields list[Tensor(C,H,W)].
             # For ONNX we need a single batched tensor [B,C,H,W].
             if len(inputs) == 0:
                 raise ValueError("Empty input list/tuple for ONNX session.")
             if isinstance(inputs[0], torch.Tensor):
                 val = torch.stack(list(inputs), dim=0).cpu().numpy()
             else:
                 val = np.stack(list(inputs), axis=0)
             ort_inputs = {self.input_names[0]: val}
        else:
            raise ValueError(f"Unsupported input type for ONNX session: {type(inputs)}")
            
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
            
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                # Handle dynamic shapes if necessary (using max profile) - simplistic approach here:
                # If shape has -1, we might need to set it. For now assuming static or handled.
                
                size = trt.volume(shape) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                # Append to bindings list (legacy) or address map (v3)
                # For execute_async_v2 we usually pass a list of pointers.
                # However, bindings order must match engine binding indices.
                # Since we iterate by index, this should be correct.
                self.bindings.append(int(device_mem))
                
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                    
        except ImportError:
            raise ImportError("tensorrt or pycuda is not installed.")

    def predict(self, inputs):
        # Copy input data to host buffer
        # Simplistic assumption: single input
        if isinstance(inputs, torch.Tensor):
            batched = inputs
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) == 0:
                raise ValueError("Empty input list/tuple for TensorRT session.")
            # stack list[Tensor(C,H,W)] -> [B,C,H,W]
            batched = torch.stack(list(inputs), dim=0)
        else:
            raise ValueError(f"Unsupported input type for TensorRT session: {type(inputs)}")

        np.copyto(self.inputs[0]['host'], batched.cpu().numpy().ravel())

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
