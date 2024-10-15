<p align="center">
    <img src="./assets/logo.png">
</p>

# Over-The-Air-Signatures-Authentication

<p align="justify">
Over-The-Air-Signatures-Authentication is a conceptual system for an engineering thesis. The system includes scripts for collecting signature data, extracting features, and applying both custom and standard machine learning classifiers from scikit-learn. These classifiers are used to compare signatures after the feature extraction process. The project will be developed further as needed.
</p>

## Table of contents

1. **[Description](#description)**
2. **[Technologies](#technologies)**
3. **[Quick start](#quick-start)**
4. **[Author](#author)**

## Description

<p align="justify"> 
The system includes scripts for collecting signature data, extracting features, and applying both custom and standard machine learning classifiers from scikit-learn.
</p>

## Technologies

Project uses following languages and technologies
* Python 3.10.12
* Mediapipe fork for GPU support https://github.com/riverzhou/mediapipe
* Ubuntu 22.04.1

## Development
### Quick start
#### Setup project locally

1. Clone the repository:

   ```
   git clone https://github.com/b4rt4s/Over-The-Air-Signatures-Authentication.git
   ```

2. Change directory:

   ```
   cd Over-The-Air-Signatures-Authentication
   ```

3. Create new virtual environment:

   ```
   python -m venv venv
   ```

4. Activate environment:

   ```
   source /venv/bin/activate
   ```

5. Install the required modules:

   ```
   python -m pip install -r requirements.txt
   ```

#### Install mediapipe only with CPU support (worse signature collection)

1. Activate previous environment:

   ```
   source /venv/bin/activate
   ```

2. Install mediapipe from pip:

   ```
   python -m pip install mediapipe
   ```

#### Install mediapipe only with GPU support (better signature collection)

1. Install mediapipe from https://github.com/riverzhou/mediapipe

> [!WARNING]
> Works only on Ubuntu 22.04  
> Install it globally (not in venv)

2. Activate previous environment:

   ```
   source /venv/bin/activate
   ```

3. Change directory:

   ```
   cd /usr/lib/x86_64-linux-gnu/mediapipe/dist
   ```

4. Install MediaPipe from a .whl file:

   ```
   python -m pip install mediapipe-0.10.1-cp310-cp310-linux_x86_64.whl
   ```

#### Run signatures collecting

> [!WARNING]
> Install mediapipe before you start collecting signatures!

1. Change directory:

   ```
   cd signatures-collecting
   ```

2. Run signatureCollecting.py:

   ```
   python3 signatureCollecting.py
   ```

> [!WARNING]
> If ImportError: cannot import name 'python' from 'mediapipe.tasks.python'  
> There is a solution: https://github.com/google-ai-edge/mediapipe/issues/4657

> [!WARNING]
> If qt.qpa.events.reader: [heap] info appears  
> Paste export QT_LOGGING_RULES="qt.qpa.events.reader=false" in the console

#### Run features extracting

1. Change directory:

   ```
   cd features-extracting
   ```

2. Install Tkinter module:

   ```
   sudo apt-get install python3-tk
   ```

3. Run SignaturePointsProcessing.py:

   ```
   python3 SignaturePointsProcessing.py
   ```

## Author

[@b4rt4s](https://github.com/b4rt4s)
