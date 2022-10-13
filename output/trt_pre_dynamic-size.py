"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

CONF_THRESH = 0.2
IOU_THRESHOLD = 0.6

alphabet = {}
with open("char_QRCode.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i,line in enumerate(lines):
    alphabet[i] = line.strip()
print("load dict: ", len(alphabet))

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def decodeOne(data):
    #t,c
    res = []
    scores = []
    # print(data)
    #print("--------------")
    last_idx = -1

    for i in range(len(data)):
        idx = np.argmax(data[i])
        # print(data[i])
        # softmax_v = softmax(data[i])
        # print(softmax_v)
        score = np.max(data[i])
        #print(idx)
        if len(res)==0:
            if idx!=0:
                res.append(idx)
                scores.append(score)
        else:
            if idx!=last_idx and idx!=0:
                res.append(idx)
                scores.append(score)
        last_idx = idx
    #print(res)
    score = 0
    if len(scores)>0:
        score = np.mean(scores)

    chars = ''.join([alphabet[x] for x in res])
    return chars,np.array(res), score


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # host_inputs = []
        # cuda_inputs = []
        # host_outputs = []
        # cuda_outputs = []
        # bindings = []

        # for binding in engine:
        #     print('bingding:', binding, engine.get_binding_shape(binding))
        #     # size = trt.volume((1,1,40,360)) * engine.max_batch_size
        #     size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        #     dtype = trt.nptype(engine.get_binding_dtype(binding))
        #     # Allocate host and device buffers
        #     host_mem = cuda.pagelocked_empty(size, dtype)
        #     cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        #     # Append the device buffer to device bindings.
        #     bindings.append(int(cuda_mem))
        #     # Append to the appropriate list.

        #     # size2 = trt.volume((1,45,2267)) * engine.max_batch_size
        #     # dtype = trt.nptype(engine.get_binding_dtype(binding))
        #     # # Allocate host and device buffers
        #     # host_mem2 = cuda.pagelocked_empty(size2, dtype)
        #     # cuda_mem2 = cuda.mem_alloc(host_mem2.nbytes)
        #     # bindings.append(int(cuda_mem2))

        #     if engine.binding_is_input(binding):
        #         self.input_w = engine.get_binding_shape(binding)[-1]
        #         self.input_h = engine.get_binding_shape(binding)[-2]
        #         host_inputs.append(host_mem)
        #         cuda_inputs.append(cuda_mem)
        #     else:
        #         host_outputs.append(host_mem)
        #         cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        # self.host_inputs = host_inputs
        # self.cuda_inputs = cuda_inputs
        # self.host_outputs = host_outputs
        # self.cuda_outputs = cuda_outputs
        # self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        # context.active_optimization_profile = 0

        engine = self.engine

        dtype = np.float32


        
        for i, image_raw in enumerate(raw_image_generator):
            bindings = []
            host_inputs = []
            cuda_inputs = []
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            self.input_h,self.input_w = input_image.shape[:2]
            context.set_binding_shape(0, (1, 1, 40, input_image.shape[1]))
            # batch_image_raw.append(image_raw)
            # batch_origin_h.append(origin_h)
            # batch_origin_w.append(origin_w)
            # np.copyto(batch_input_image[i], input_image)
            size1 = trt.volume((1,1,40,input_image.shape[1])) * engine.max_batch_size
            host_mem1 = cuda.pagelocked_empty(size1, dtype)
            cuda_mem1 = cuda.mem_alloc(host_mem1.nbytes)
            bindings.append(int(cuda_mem1))
            
            host_inputs.append(host_mem1)
            cuda_inputs.append(cuda_mem1)    

            ##output
            host_outputs = []
            cuda_outputs = []

            size2 = trt.volume((1,input_image.shape[1]//8,len(alphabet))) * engine.max_batch_size
            host_mem2 = cuda.pagelocked_empty(size2, dtype)
            cuda_mem2 = cuda.mem_alloc(host_mem2.nbytes)

            host_outputs.append(host_mem2)
            cuda_outputs.append(cuda_mem2)
            bindings.append(int(cuda_mem2))

            # Do image preprocess
            batch_image_raw = []
            batch_origin_h = []
            batch_origin_w = []

            batch_input_image = np.reshape(input_image, (1,1,input_image.shape[0],input_image.shape[1]))
            batch_input_image = np.ascontiguousarray(batch_input_image)

            # print(host_inputs[0].shape,batch_input_image.shape)
            # bb
            # Copy input image to host buffer
            # print(batch_input_image)
            np.copyto(host_inputs[0], batch_input_image.ravel())
            start = time.time()
            # Transfer input data  to the GPU.
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            # Run inference.
            # print('1')
            # context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            # print('2')
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            # Synchronize the stream
            stream.synchronize()
            end = time.time()
            # Remove any context from the top of the context stack, deactivating it.
            self.ctx.pop()
            # Here we use the first row of output in that batch_size = 1
            output = host_outputs[0]
            # print(output, output.shape)
            output = np.reshape(output, (-1,len(alphabet)))

            # Do postprocess
            # print(np.sum(output))
            # b
            res = decodeOne(output)
            print(res, end - start)
            #b
        return res, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        if c==3:
            img = cv2.cvtColor(image_raw,cv2.COLOR_BGR2GRAY)

        # Calculate widht and height and paddings
        resize_h = 40
        h,w = img.shape[:2]
        resize_w = int(w*resize_h/h)#+1
        img = cv2.resize(img, (resize_w,resize_h))

        data = img.reshape( 1, 1, img.shape[ 0], img.shape[ 1])
        image = img.astype(np.float32)
        # Normalize to [0,1]
        image = image/255.0-0.5
        # HWC to CHW format:
        # image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        # image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)

        # print(image.shape)
        # print(image)
        # b
        return image, image_raw, h, w



class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        res, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        # for i, img_path in enumerate(self.image_path_batch):
        #     parent, filename = os.path.split(img_path)
        #     save_name = os.path.join('output', filename)
        #     # Save image
        #     cv2.imwrite(save_name, batch_image_raw[i])
        # print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))



if __name__ == "__main__":
    # load custom plugin and engine

    engine_file_path = "model.engine"



    yolov5_wrapper = YoLov5TRT(engine_file_path)
    try:
        print('batch size is', yolov5_wrapper.batch_size)
        
        image_dir = "imgs"
        image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)

        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(yolov5_wrapper, batch)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
