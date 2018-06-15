
import tensorflow as tf
from resnet152 import ResNet152
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import numpy as np
import timeit as t
import base64
import json
from PIL import Image, ImageOps
from io import BytesIO
import logging

number_results = 3
logger = logging.getLogger("model_driver")

def _base64img_to_numpy(base64_img_string):
    decoded_img = base64.b64decode(base64_img_string)
    img_buffer = BytesIO(decoded_img)
    imageData = Image.open(img_buffer).convert("RGB")
    img = ImageOps.fit(imageData, (224, 224), Image.ANTIALIAS)
    img = image.img_to_array(img)
    return img

def create_scoring_func():
    """ Initialize ResNet 152 Model 
    """   
    start = t.default_timer()
    model = ResNet152(weights='imagenet')
    end = t.default_timer()
    
    loadTimeMsg = "Model loading time: {0} ms".format(round((end-start)*1000, 2))
    logger.info(loadTimeMsg)
    
    def call_model(img_array):
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array)
        preds = decode_predictions(preds, top=number_results)[0]       
        return preds
    
    return call_model       

def get_model_api():
    logger = logging.getLogger("model_driver")
    scoring_func = create_scoring_func()
    
    def process_and_score(inputString):
        """ Classify the input using the loaded model
        """
        start = t.default_timer()

        responses = []
        base64Dict = json.loads(inputString) 
        for k, v in base64Dict.items():
            img_file_name, base64Img = k, v 
        img_array = _base64img_to_numpy(base64Img)
        preds = scoring_func(img_array)
        resp = {img_file_name: preds}
        responses.append(resp)

        end = t.default_timer()
        
        logger.info("Predictions: {0}".format(responses))
        logger.info("Predictions took {0} ms".format(round((end-start)*1000, 2)))
        return (responses, "Computed in {0} ms".format(round((end-start)*1000, 2)))
    return process_and_score

def version():
    return tf.__version__