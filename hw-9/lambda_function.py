import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

onnx_model_path = "hair_classifier_empty.onnx"

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    X = np.array(img)
    X = X.reshape(1, img.height, img.width, 3)
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std

    return X.astype(np.float32)

# creaete ONNX Runtime session
session = ort.InferenceSession(
    onnx_model_path, 
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# predict function
def predict(img_url):
    img = download_image(img_url)
    img = prepare_image(img, target_size=(200, 200))
    X = preprocess(img)
    result = session.run([output_name], {input_name: X})
    value = result[0][0][0]
    
    cls = "curly" if value < 0.5 else "straight"
    return {"result": cls, "probability": float(value)}

def lambda_handler(event, context):
    img_url = event['url']
    prediction = predict(img_url)
    
    return {
        'statusCode': 200,
        'body': str(prediction)
    }