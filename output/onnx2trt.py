#!/usr/bin/python3
# Build TensorRT engine from ONNX saved model and serialize engine to file 

import tensorrt as trt
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_path, using_half,engine_file,dynamic_input):
    trt.init_libnvinfer_plugins(None, '')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        config.max_workspace_size = 2**20#GiB(1)
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        ##增加部分
        if dynamic_input:
            profile = builder.create_optimization_profile();
            profile.set_shape("input1", (1,1,40,40), (1,1,40,360), (1,1,40,1000)) 
            config.add_optimization_profile(profile)
        #加上一个sigmoid层
        # previous_output = network.get_output(0)
        # network.unmark_output(previous_output)
        # sigmoid_layer=network.add_activation(previous_output,trt.ActivationType.SIGMOID)
        # network.mark_output(sigmoid_layer.get_output(0))

        engine =  builder.build_engine(network, config) 

        with open(engine_file, "wb") as f:
            f.write(engine.serialize())

build_engine("model.onnx", False, "model.engine", True)
