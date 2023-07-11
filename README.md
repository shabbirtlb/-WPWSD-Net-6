# WPWSD-Net-6

WPWSD-Net-6 is a 2D-CNN based classification software developed for the purpose of detecting the presence of Wolff Parkinson White Syndrome in patients.

## Objective

The primary objectives of WPWSD-Net-6 would be to provide: 

Achieving high accuracy: WPWSD-Net-6 is designed to identify WPW patterns with high sensitivity and specificity, and minimize the occurrence of false negatives and false positives.

Reducing diagnosis time: WPWSD-Net-6 is designed to process ECG recordings quickly and efficiently, reducing the time required for diagnosis to improve patient outcomes and reduce the workload of healthcare professionals.

Improving accessibility: WPWSD-Net-6 is user-friendly and accessible to non-experts  particularly in areas with limited access to medical professionals or specialized equipment.

Enhancing reproducibility: WPWSD-Net-6 is designed to produce consistent and reproducible results, reducing inter-observer variability and improving the reliability of WPW diagnosis.

Incorporating robustness and adaptability: WPWSD-Net-6 is able to handle a diverse range of ECG recordings from different populations, and adapt to changes in ECG acquisition protocols and hardware. 

Advancing the state-of-the-art: The development of WPWSD-Net-6 using a 2D CNN can advance the state-of-the-art in diagnosing WPW syndrome and potentially inform future research in cardiology and medical imaging.

## Methodology

Data collection and preprocessing: Dataset should be preprocessed to remove noise, artifacts, and baseline wander.

Data augmentation: Data augmentation techniques such as flipping, rotating, and scaling is applied to the ECG recordings to increase the diversity of the dataset.

Model architecture design: A 2D CNN architecture is designed and optimized for WPW detection including several convolutional layers followed by pooling,dropout and fully connected layers.

Model training: The designed model should be trained on the preprocessed and augmented dataset involving minimizing the loss function using backpropagation and stochastic gradient descent.

Model evaluation: The trained model should be evaluated on a test dataset to assess its accuracy, sensitivity, and specificity used to fine-tune the model.

Model deployment: The final step is to deploy the trained model as a software application that can accept new ECG recordings and provide a diagnosis of WPW syndrome.

## Installation

Install all the required libraries for the software by executing the following commands in Command Prompt: 

```bash
1. pip freeze > requirements.txt
2. pip install -r requirements.txt
```

## Usage

Python

Note : Create a folder titled "Uploads" inside the repository after cloning it locally.

## Results
Evaluation Parameters for WPWSD-Net-6 software:

Accuracy: 96.06%
F1 score: 91.3%
Precision:97.1%
Recall :86.25%



## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
