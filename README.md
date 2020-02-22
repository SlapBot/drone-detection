
# Drone Detection (Dronacharya)

A deep learning neural net model to detect drone/drones from a given picture using Using Fast R-CNN architecture via Keras-Retinanet Implementation. (Dataset and Pre-Trained model provided) 

Live at: http://dronacharya.slapbot.me/

- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
    - [Pre-Requisites](#pre-requisites)
    - [Prediction](#prediction)
	    - [System Setup](#system-setup)
	    - [Application Setup](#application-setup)
	    - [Usage](#usage)
    - [Training](#system-setup)
	    - [GCP Instance](#gcp-instance)
	    - [Graphic Card Drivers Setup](#graphic-card-drivers-setup)
		    - [Install CUDA 8.0](#install-cuda-8.0)
		    - [Install cuDNN 6.0](#install-cudnn-6.0)
	    - [Training The Network](#training-the-network)
- [Usage](#usage)
    - [Advanced API](#advanced-api)
- [Results](#results)
    

## Motivation

- Me and my partner [Nilesh](https://github.com/nileshtrivedi) participated in a Hackathon called, [MoveHack](https://pib.gov.in/newsite/PrintRelease.aspx?relid=181379) which had one of the problem statement of Drone and UAV traffic management.

- One of the key challenges of the problem statement was to detect any UAV or Drone from a given image.

- I took the challenge by researching online of different techniques of detecting objects from a given picture, and with a prior of experience of using Fast R-CNN architecture in my workplace, I just went with it to see how it fares against drone detection.

![](https://github.com/SlapBot/drone-detection/blob/master/screenshots/1.gif)


## Features

- A pre-trained model is included in the repository ready to be used out of the box for drone detections.
- Multiple drones can be detected from an image.
- Dataset used to train the model with clear instructions are provided in the case you'd want to train over a larger dataset.
- Simple Intuitive API is provided to help in prediction task with full control over tolerance of detecting drones.
- The entire source code is well documented and uses type hinting for more stability.
- The installation instructions are separated into two categories depending on your use-case: 
	- Training: Well documented instructions from scratch to getting the model trained.
	- Prediction: Specific instructions to simply use pre-trained model right off the bat and go with the workflow.

## Installation

Installation is divided into two parts:
- Prediction
	- You'd want to use pre-trained model to detect drones in a given image.
- Training
	- You'd want to train the model with larger dataset/fine tine hyper-parameter, etc.

### Pre-requisites

1. Python3
2. pip
3. virtualenv


## Prediction

### System Setup

1. Update the package index: `sudo apt-get update`
2. Install Additional development libraries: `sudo apt-get install python3-dev python3-pip libcupti-dev`
3. Install Additional system libraries: `sudo apt-get install libsm6 libxrender1 libfontconfig1`
4. Download the pre-trained model to the `Trained-Model` directory under name: `drone-detection-v5.h5` from this link: https://drive.google.com/open?id=1nRMPUQcW9U6E3WjlP751s_77U9a0R5A9

### Application Setup

1. Clone the repo: `git clone https://github.com/slapbot/drone-detection`
2. Cd into the directory: `cd drone-detection`
3. Create a virtual-env for python: `python -m venv drone-detection-env`
4. Activate the virtual-env: `source drone-detection-env/bin/activate`
5. Upgrade your pip to latest version: `pip install --upgrade pip`
6. Install numpy: `pip install numpy==1.17.0`
7. Install the application dependencies: `pip install -r requirements.txt`
	- This will install Tensorflow CPU, if you want to install GPU version, swap out tensorflow with tensorflow-gpu in `requirements.txt`
8. Run `python evaluate.py` to detect drones from one of the test image saved in the `Dataset` folder.

### Usage

As you can see below the API is super intuitive and self-explaining to use.
```
from core import Core  

c = Core()  
  
image_filename = c.current_path + "/DataSets/Drones/testImages/351.jpg"  
image = c.load_image_by_path(image_filename)  
  
drawing_image = c.get_drawing_image(image)  
  
processed_image, scale = c.pre_process_image(image)  
  
c.set_model(c.get_model())  
boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)  
  
detections = c.draw_boxes_in_image(drawing_image, boxes, scores)  
  
c.visualize(drawing_image)
```

## Training

###  GCP Instance  
  
  Create a virtual machine with these specifications. (You're open to use any other host provider or VM, its just what I did in the process.)
  
```  
CPU 8 core 30 GB memory  
server location: us-west1-b  
GPU 1 Nvidia Tesla K80  
```  
  
### Graphic Card Drivers Setup  
  
#### Install CUDA 8.0  
  
1. Update Repositories: `sudo-apt get update`  
2. Create an installation shell script: `nano install_cuda.sh`  
```  
#!/bin/bash  
echo "Checking for CUDA and installing."  
# Check for CUDA and try to install.  
if ! dpkg-query -W cuda; then  
  # The 16.04 installer works with 16.10.  
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb  
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb  
  apt-get update  
  # apt-get install cuda -y  
  sudo apt-get install cuda-8-0  
fi  
```  
3. Login as root user: `sudo su`  
4. Install Cuda 8.0: `./install_cuda.sh`  
5. Verify the installation: `nvidia-smi`  
6. Export required env variables:  
```  
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc  
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc  
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc  
source ~/.bashrc  
```  
  
#### Install cuDNN 6.0  
  
1. Download cuDNN from Nvidia Main website (6.0 version), in my case I saved it in private repo. `git clone https://slapbot@bitbucket.org/slapbot/cudnn.git`  
2. Cd in directory: `cd cudnn`  
3. Install .deb package: `sudo dkpg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb`  
  
### Python Setup  
  
1. Install python packages: `sudo apt-get install python3-dev python3-pip libcupti-dev`  
2. Install Tensorflow GPU 1.4: `pip3 install --upgrade tensorflow-gpu==1.4.0`  
3. Verify Tensorflow Installation: `python3 -c "import tensorflow as tf; print(tf.__version__)"`  
  
### Drone Detection Setup  
  
1. Clone the repo: `git clone https://github.com/slapbot/drone-detection`
2. Cd into the directory: `cd drone-detection`  
3. Clone keras-retinanet repo: `git clone https://github.com/fizyr/keras-retinanet.git`  
4. Cd in keras-retinanet repo: `cd keras-retinanet`  
5. Install package: `pip3 install . --user`  
6. Install repository wide deps: `python3 setup.py build_ext --inplace`  
7. Return back to main directory: `cd ..`  
  
### Training the network
  
1. Generate annotations, labels and validation annotations: `python3 data_preparation.py`  
2. Install More packages if necessary - (want to visualise):  
```  
pip install opencv-python
pip install Pillow
```  
3. Train the model using:  
```  
python3 keras-retinanet/keras_retinanet/bin/train.py csv annotations.csv classes.csv --val-annotations=validation_annotations.csv  
```    
4. Convert the trained model to inference model:  
```  
python3 keras-retinanet/keras_retinanet/bin/convert_model.py resnet50_csv_05.h5 resnet50_csv_05_inference.h5
```
5. Now simply copy back the model to `Trained-Model` directory and follow the prediction instructions to get started with predicting!

## Usage

The API is super straightforward and intuitive to understand and consume, 
taking a look at the `evaluate.py` should give you a rough understanding of its functioning.

```
from core import Core  

c = Core()  
  
image_filename = c.current_path + "/DataSets/Drones/testImages/351.jpg"  
image = c.load_image_by_path(image_filename)  
  
drawing_image = c.get_drawing_image(image)  
  
processed_image, scale = c.pre_process_image(image)  
  
c.set_model(c.get_model())  
boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)  
  
detections = c.draw_boxes_in_image(drawing_image, boxes, scores)  
  
c.visualize(drawing_image)
```


### Advanced API

- `boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)`: returns you the boxes, scores and labels of the objects found (in our case labels are just used to do binary classification so there are only two labels.)
- Each item of box has an associated score, so `boxes[0]` co-relates with `scores[0]` and so on, depending on the score, you can say whether its a drone or not, after experimenting, I've figured that `0.5` is a good threshold value for tolerance.

## Results

Prototyped in  [MoveHack](http://pib.gov.in/newsite/PrintRelease.aspx?relid=181379)  - Was selected as top 10 overall solutions across all challenge themes among  [7,500 individuals and 3,000 teams](https://www.thehindubusinessline.com/info-tech/7500-individuals-register-for-movehack-niti-aayogs-global-mobility-hackathon/article24736986.ece) that globally competed for Hackathon.

Won the cash prize of ₹10,00,000 and received an invitation to attend the  [Global Mobility Summit 2018](http://movesummit.in/about.php)  at Vigyan Bhawan, Delhi by NITI AAYOG to meet major CEOs across automobiles, aviation, mobility organisations and receive the award by Prime Minister of India, Narendra Modi.
