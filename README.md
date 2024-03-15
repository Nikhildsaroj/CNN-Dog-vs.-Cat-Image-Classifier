

# CNN Dog vs. Cat Image Classifier

### Description:
This repository hosts a project that classifies images of dogs and cats using a convolutional neural network (CNN). Leveraging the Kaggle Dogs vs. Cats dataset, comprising over 25,000 images, the project employs Python code and TensorFlow/Keras libraries for robust model development and training.

### Key Details:
### **Platform to Use:** Google Colab 
- **Dataset:** The Kaggle Dogs vs. Cats dataset features an extensive array of dog and cat images, meticulously categorized into train, validation, and test subsets. These subsets serve as the cornerstone for training, validating, and evaluating the CNN model.
- **Model:** Employing the VGG16 model architecture, a renowned deep convolutional neural network crafted by the Visual Geometry Group at the University of Oxford, our project harnesses 16 layers, encompassing both convolutional and fully connected layers.
- ![image](https://github.com/Nikhildsaroj/CNN-Dog-vs.-Cat-Image-Classifier/assets/148480961/fcdbb34e-dbbe-438f-a6e3-d470d804b97a)

- **Training:** Our CNN model undergoes rigorous training using the training and validation subsets. Through meticulous parameter optimization and early stopping mechanisms, we ensure that our model learns to minimize classification loss over multiple epochs, while steering clear of the treacherous waters of overfitting.
- ![image](https://github.com/Nikhildsaroj/CNN-Dog-vs.-Cat-Image-Classifier/assets/148480961/7b8215b1-27ee-41a2-bc8d-4d81d5f70149)

- **Evaluation:** To gauge the prowess of our trained CNN model, we subject it to the ultimate litmus test—the test subset of the dataset. Armed with metrics such as accuracy, we meticulously assess its ability to discern between images of dogs and cats.
- ![image](https://github.com/Nikhildsaroj/CNN-Dog-vs.-Cat-Image-Classifier/assets/148480961/0c150a4c-f6eb-4b7e-8e32-b27305eee58e)

- ![image](https://github.com/Nikhildsaroj/CNN-Dog-vs.-Cat-Image-Classifier/assets/148480961/6a69bb7e-bc93-450e-815a-44d14da3471c)



### Repository Contents:
- **Jupyter Notebook:** Our primary notebook contains Python code for data preprocessing, model architecture definition, training, and evaluation. Extensive explanations and comments are provided to ensure a seamless journey through the code implementation.
- **Dataset Files:** The Kaggle Dogs vs. Cats dataset files are included within the repository or can be sourced directly from Kaggle: (https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- Comprehensive instructions for dataset loading and preprocessing are available within the notebook.

### Instructions:
Embark on an immersive exploration of the CNN Dog vs. Cat Image Classifier project and delve into the nitty-gritty of model development. Clone this repository and execute the provided Jupyter Notebook. Follow the detailed instructions within to load and preprocess the dataset, define and train the CNN model, and evaluate its performance.

### Kaggle API Integration for Google Colab:
To seamlessly integrate the Kaggle API into your Google Colab environment and download the Dogs vs. Cats dataset, follow these steps:
- **Create a Kaggle account** and procure an API token from your account settings.
- **Upload the kaggle.json API token file** to your Google Colab environment.
- **Optimize your runtime environment** by switching to T4 GPU for expedited processing.
- Utilize the Kaggle API command **!kaggle datasets download -d salader/dogs-vs-cats** within Colab to download the dataset directly.
- Execute the provided code in the repository to preprocess the dataset, define and train the CNN model, and evaluate its performance.

### Model Performance:
- **Training Accuracy:** 98.60%
- **Validation Accuracy:** 93.84%
- ![image](https://github.com/Nikhildsaroj/CNN-Dog-vs.-Cat-Image-Classifier/assets/148480961/7cf4eef3-0012-4351-9233-c0ef74edfa4b)


