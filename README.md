# Percentage-of-Food-Images
Here I develop 3 different approaches based on CNN to calculate the percentage of food (pixels) in images.

### [Models](models.py)
We can find the different models>
 - ResNet
 - VGG
 - Ad hoc
---

### [Config](config.py)
File containing the different variables needed to run the project (images folder. epochs, batch size...)

---
### [Training](training.py)

Here we load the data, create the different keras generators and callbacks and train and evaluate the selected model.
