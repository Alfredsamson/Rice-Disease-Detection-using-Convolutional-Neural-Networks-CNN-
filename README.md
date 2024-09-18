# Rice-Disease-Detection-using-Convolutional-Neural-Networks-CNN-

This repository contains a Python notebook that implements a convolutional neural network (CNN) to detect diseases in rice plants using image data. The project uses TensorFlow and Keras to train a model on a dataset of rice leaf images, classifying them into various disease categories.

## Dataset
The dataset used in this project is a collection of rice leaf images, classified into six disease categories. The dataset is loaded from the following directory: /kaggle/input/rice-disease-dataset/Rice_Leaf_AUG.

The dataset contains 3,829 images belonging to the following categories:

### Bacterial Leaf Blight
### Blast
### Brown Spot
### Healthy
## Project Structure
## Data Preprocessing:

The images are preprocessed using TensorFlow's image_dataset_from_directory function.
Image augmentation is used to improve the model's generalization.
## Model Architecture:

A CNN architecture is defined using TensorFlow and Keras.
It includes multiple convolutional layers, max-pooling layers, and dense layers for classification.
Model Training:

The model is trained using 50 epochs, with a batch size of 20, and an image size of 128x128 pixels.
Accuracy and loss metrics are tracked during training.
Evaluation:

The model's performance is evaluated using a confusion matrix and other relevant metrics.
## Visualization:

The training and validation results are visualized using Matplotlib to observe trends in accuracy and loss.
Requirements
The following libraries are required to run the notebook:

### numpy
### pandas
### tensorflow
### matplotlib
sklearn
You can install the dependencies via pip:

bash
Copy code
pip install numpy pandas tensorflow matplotlib scikit-learn
How to Run
Clone the repository and navigate to the directory:
bash
Copy code
git clone https://github.com/yourusername/rice-disease-detection.git
cd rice-disease-detection
Install the dependencies listed above.
Open the Jupyter Notebook:
bash
Copy code
jupyter notebook rice-disease-detection-using-cnn-python.ipynb
Run all cells in the notebook to load the dataset, train the model, and evaluate its performance.
Results
The model is able to classify the rice leaf diseases with a certain level of accuracy (exact metrics will be displayed in the output section of the notebook).
## Future Improvements
Fine-tuning of the model architecture for better performance.
Experimenting with different learning rates and optimizers.
Expanding the dataset to include more disease categories and image samples.
