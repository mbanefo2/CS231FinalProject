# CS231_final_project
This repo contains code for CS231A Final Project Trajectory detection and boundary estimation for badminton shuttlecock using 3D position estimation

To setup and run inference on our YOLOv5 trained network. Use the following

```
from roboflow import Roboflow
rf = Roboflow(api_key="U2gYzbPl5B8gh3WPq7P3")
project = rf.workspace().project("finding-shuttlecock")
model = project.version(1).model

# infer on a local image
print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
```

After that you can put your innference outputs into the main.py file along with the paths to the image you want to test on

Then run the following command

```
python3 main.py
```
