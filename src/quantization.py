import torch
import onnx
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], dynamic_axes=None):
    """
    Exports a PyTorch model to ONNX format.
    """
    logger.info(f"Exporting model to {onnx_path}...")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17 
        )
        logger.info("✅ ONNX export successful.")
        
        # Verify the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model verified.")
        
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        raise e

def quantize_onnx_model(onnx_path, quantized_path):
    """
    Quantizes an ONNX model using dynamic quantization (UInt8).
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        
        logger.info(f"Quantizing model to {quantized_path}...")
        
        # Pre-process the model to improve shape inference
        preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
        try:
            quant_pre_process(onnx_path, preprocessed_path)
            logger.info("✅ ONNX pre-processing successful.")
            input_model_path = preprocessed_path
        except Exception as e:
            logger.warning(f"⚠️ ONNX pre-processing failed: {e}. Proceeding with original model.")
            input_model_path = onnx_path

        quantize_dynamic(
            input_model_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        logger.info("✅ ONNX quantization successful.")
        
    except ImportError:
        logger.error("❌ onnxruntime is not installed. Please install it to use quantization.")
    except Exception as e:
        logger.error(f"❌ ONNX quantization failed: {e}")
        raise e

def build_tensorrt_engine(onnx_path, engine_path, fp16=True, input_shapes=None):
    """
    Builds a TensorRT engine from an ONNX file.
    
    Args:
        onnx_path: Path to ONNX model.
        engine_path: Path to save TensorRT engine.
        fp16: Enable FP16 precision.
        input_shapes: Dict mapping input names to (min_shape, opt_shape, max_shape) tuples.
                      Example: {'input': ((1, 3, 224, 224), (8, 3, 224, 224), (8, 3, 224, 224))}
    """
    try:
        import tensorrt as trt
        
        logger.info(f"Building TensorRT engine to {engine_path}...")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        
        # Create network definition with explicit batch
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        config = builder.create_builder_config()
        
        # Set memory pool limit (e.g., 1GB for workspace)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision.")

        # Create Optimization Profile if input_shapes provided
        if input_shapes:
            profile = builder.create_optimization_profile()
            for name, (min_shape, opt_shape, max_shape) in input_shapes.items():
                logger.info(f"Adding optimization profile for {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            
        # Parse ONNX
        if not parser.parse_from_file(onnx_path):
            logger.error("❌ Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None
        
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine:
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
            logger.info("✅ TensorRT engine built and saved.")
        else:
            logger.error("❌ Failed to build TensorRT engine.")
            
    except ImportError:
        logger.error("❌ tensorrt python bindings not found. Please install tensorrt.")
    except Exception as e:
        logger.error(f"❌ TensorRT build failed: {e}")
        raise e
