# Webcam ArcaneGAN
Simple wrapper around [ArcaneGAN](https://github.com/Sxela/ArcaneGAN) that allows to use it as a virtual webcam.
Wrapper uses default system webcam as input channel and one of system virtual webcams as output channel (Check [PyVirtualCam](https://github.com/letmaik/pyvirtualcam) doc for more info).

## Requirments
### Hardware
Tested on Nvidia 3070Ti. Will work with other GPUs with lower performance. Having GPU is highly recommended.

### Libraries
* [PyTorch](https://pytorch.org/)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [PyVirtualCam](https://pypi.org/project/pyvirtualcam/)

## Usage
1. Setup required libraries
2. Download [ArcaneGAN checkpoint](https://github.com/Sxela/ArcaneGAN/releases/tag/v0.4)
3. Run `main.py`