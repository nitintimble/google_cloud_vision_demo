import os
import io
import pandas as pd
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import prepare_image_local, prepare_image_web, draw_boundary, draw_boundary_normalized

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='facelive-420408-7c369420f419.json'
client = vision.ImageAnnotatorClient()

# image_path = 'images/Sachin-Tendulkar.jpg'
# with io.open(image_path, 'rb') as image_file:
#     content = image_file.read()
# image = vision.Image(content=content)


# image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Sachin-Tendulkar_%28cropped%29.jpg/220px-Sachin-Tendulkar_%28cropped%29.jpg'
# imageweb = vision.Image()
# imageweb.source.image_uri = image_url

# image_path = 'images/Sachin-Tendulkar.jpg'
image_path = 'images/kirana_1.jpg'
image1 = prepare_image_local(image_path=image_path)
va = VisionAI(client=client, image=image1)
label_detecions = va.label_detection()

df = pd.DataFrame(label_detecions)
print(df)
