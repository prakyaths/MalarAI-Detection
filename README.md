# MalarAI-Detection
This project implements a Convolutional Neural Network (CNN) based on the LeNet architecture to detect malaria from cell images.

Overview
This project implements a convolutional neural network (CNN) based on the LeNet architecture to classify whether a given blood cell image is infected with the Plasmodium parasite (responsible for Malaria) or not. The model is trained on a publicly available dataset of microscopic images of blood cells, where the cells are either parasitized or uninfected.

Project structure:

| Directory/File                   | Description                                     |
|----------------------------------|-------------------------------------------------|
| `data/`                          | Directory to store the malaria dataset          |
| `models/`                        | Directory to save trained models                |
| `src/`                           | Source code for model building and training     |
| ├── `malaria_detection.ipynb`    | Main Colab notebook for the project             |
| `results/`                       | Directory to save evaluation metrics and plots  |
| `README.md`                      | Project documentation (this file)               |

Dataset
The dataset used for this project is the Malaria Cell Images Dataset from the National Institutes of Health (NIH). It contains over 27,000 images, with roughly half of the images showing parasitized cells and the other half showing healthy (uninfected) cells.

Parasitized cells: Cells infected with Plasmodium parasites.
Uninfected cells: Healthy cells with no infection.

Model Architecture
The model is based on the classic LeNet-5 CNN architecture, consisting of:

Input Layer: 64x64 RGB images
Convolutional Layers: Two 2D convolutional layers with ReLU activations
Pooling Layers: Two max-pooling layers after the convolutional layers
Fully Connected Layers: Two fully connected layers
Output Layer: Sigmoid layer for binary classification (Parasitized vs. Uninfected)

Here's the architecture in more detail:

| Layer               | Description                                  |
|---------------------|----------------------------------------------|
| Conv2D              | 6 filters, kernel size 5x5, ReLU activation  |
| MaxPooling2D        | Pool size 2x2                                |
| Conv2D              | 16 filters, kernel size 5x5, ReLU activation |
| MaxPooling2D        | Pool size 2x2                                |
| Flatten             | Reshapes the 2D matrices into 1D vectors     |
| Fully Connected (FC)| 100 units, ReLU activation                   |
| Fully Connected (FC)| 10 units, ReLU activation                    |
| Output (Softmax)    | 1 unit (Parasitized/Uninfected)              |

Training
The model is trained using the Adam optimizer with a learning rate of 0.001. The loss function used is Binary cross-entropy since this is a binary classification task. The training process includes:

Data Augmentation: The dataset is augmented with horizontal flips and zooms to improve generalization.
Early Stopping: To prevent overfitting, early stopping is used based on validation loss.
Batch Size: A batch size of 32 is used for training.
Epochs: The model is trained for 20 epochs.

Evaluation
After training, the model is evaluated on the test set using standard metrics:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

Sample Results
The model achieves a classification accuracy of approximately 95% on the test set. The confusion matrix and classification report are generated to analyze performance further.

Future Improvements
Improve Accuracy: Experimenting with deeper networks such as ResNet or transfer learning approaches might help improve classification accuracy.
Hyperparameter Tuning: Optimizing hyperparameters such as learning rate, batch size, and dropout rates could enhance performance.
Better Data Augmentation: Implementing more diverse data augmentation techniques such as contrast adjustments could improve the robustness of the model.

Conclusion
This project demonstrates the application of CNNs for malaria detection using blood cell images. While the LeNet architecture provides good results, there is room for improvement through deeper models and more advanced techniques.
