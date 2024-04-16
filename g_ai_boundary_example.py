import os
import io
import pandas as pd
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import prepare_image_local, prepare_image_web, draw_boundary, draw_boundary_normalized

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='facelive-420408-7c369420f419.json'
client = vision.ImageAnnotatorClient()

# image_path = 'images/tourist-places1.png' # works
image_path = 'images/tourist-places2.png' # works
image1 = prepare_image_local(image_path=image_path)
va = VisionAI(client=client, image=image1)
landmarks = va.landmark_detection()
for landmark in landmarks:
    print(landmark.description)
    print(landmark.score)
    draw_boundary(image_file=image_path,vertices= landmark.bounding_poly, caption= landmark.description)