# OBJECT-DETECTION-SYSTEM
Project Goal: Multi-Label Image Classification The primary goal of this Python script is to build and train a deep learning model that can perform multi-label image classification. Unlike single-label classification (which identifies only one dominant object per image),




What the Code Does
This Python script is a complete workflow for training and testing a deep learning model for object recognition. It uses the popular ResNet-18 architecture, pre-trained on the ImageNet dataset, and fine-tunes it to identify up to 80 different object categories from the COCO 2017 dataset.

Here's a breakdown of the key steps:

Data Preparation: The code sets up the paths to the COCO 2017 dataset, which includes images and annotation files. The COCODataset class is a custom PyTorch Dataset that reads the images and their corresponding annotations. Unlike single-label classification (like on ImageNet, which predicts one main object per image), this code creates a multi-hot encoded label vector for each image. This vector is a list of 0s and 1s, where a 1 at a specific position indicates the presence of a particular object category (e.g., a person, a car, or a cat) in the image.

Model Setup: A ResNet-18 model is loaded with pre-trained weights. The final layer of the network, which originally had 1000 output neurons, is replaced with a new layer that has 80 neurons, one for each of the COCO categories. This modification adapts the model for the new multi-label classification task.

Training Loop: The model is trained using a standard deep learning loop. The BCEWithLogitsLoss function is used to calculate the difference between the model's predictions and the true labels. This loss function is ideal for multi-label problems because it treats each label as an independent binary classification task. The Adam optimizer is used to update the model's weights based on this loss, making it more accurate over time.

Testing and Validation: After training, the model's performance is evaluated on a separate validation set. It calculates validation accuracy by comparing the predicted probabilities (after applying a sigmoid function) against the true labels. A prediction is considered positive if its probability exceeds a threshold of 0.5.

Single Image Prediction: Finally, the predict_image function demonstrates how to use the trained model to predict the objects in a single, unseen image. It takes an image path, runs it through the model, and returns a list of the predicted object names, like "car," "person," or "bicycle." This is the real-world application of the entire process.

What the Code Predicts
The model predicts which object categories are present in an image. For example, if you give it an image of a street scene, it might predict a list of categories like:

person

car

traffic light

bus

building

It's designed to identify all relevant objects, not just the most prominent one. This is in contrast to typical single-label classification models that would only output "street scene" or "car."






DATA SET PATH= /kaggle/input/coco-2017-dataset
