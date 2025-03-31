# Neural_Networks

# Traffic Sign Classifier

This project is a deep learning-based traffic sign classifier that uses a Convolutional Neural Network (CNN) trained on a dataset of traffic signs. The model can predict traffic sign categories from images.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [GUI Application](#gui-application)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features
- Classifies images into one of 43 traffic sign categories.
- CNN-based model trained with TensorFlow/Keras.
- GUI for easy image classification.
- Model evaluation and testing scripts included.

## Dataset
This model is trained using the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset, which contains images of 43 different traffic signs.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-classifier.git
   cd traffic-sign-classifier
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the GUI
To classify an image using the GUI:
```bash
python gui_app.py
```
This will open a window where you can upload an image and get predictions.

### Command-line Prediction
To classify an image from the command line:
```bash
python test_model.py best_model.h5 path/to/image.jpg
```

## Training the Model
To train the model, run:
```bash
python traffic.py path/to/dataset [optional_model_name.h5]
```
This will train a CNN model and save it as `model.h5` if specified.

## Testing the Model
To test a trained model on an image, use:
```bash
python test_model.py best_model.h5 path/to/test/image.jpg
```

## GUI Application
A GUI built with Tkinter allows users to upload and classify traffic sign images interactively. Run:
```bash
python gui_app.py
```

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Tkinter (for GUI)

## Contributing
Feel free to submit pull requests or report issues. Contributions are welcome!

## License
This project is open-source under the [MIT License](LICENSE).
