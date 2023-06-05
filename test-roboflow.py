from roboflow import Roboflow
rf = Roboflow(api_key="3KqG1UPZmi8eBhoVjB7Q")
project = rf.workspace().project("currency-identification-smart-glasses")
model = project.version(1).model

# infer on a local image
print(model.predict("2.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())