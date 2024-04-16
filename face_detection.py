import os
import io
import pandas as pd
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import prepare_image_local, prepare_image_web, draw_boundary, draw_boundary_normalized

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='facelive-420408-7c369420f419.json'
client = vision.ImageAnnotatorClient()

# image_path = 'images/original_anil.jpg'
image_path = 'images/photo_of_photo.jpg'
image1 = prepare_image_local(image_path=image_path)
# va = VisionAI(client=client, image=image1)

response = client.face_detection(image1)

print(response)