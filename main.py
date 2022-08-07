import torch
from torchvision import transforms
import cv2
import pyvirtualcam
import os
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class ArcaneGAN:
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    t_stds = torch.tensor(stds).to(device).half()[:, None, None]
    t_means = torch.tensor(means).to(device).half()[:, None, None]
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])

    def __init__(self, model_path=f"./ArcaneGANv0.4.jit", img_size=(512, 512)):
        self.model = torch.jit.load(model_path).eval().to(device).half()
        self.img_size = img_size

    def _fit(self, img):
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def process(self, img):
        img = self._fit(img)
        transformed_image = self.img_transforms(img)[None, ...].to(device).half()

        with torch.no_grad():
            result_image = self.model(transformed_image)[0]
            output_image = result_image.mul(self.t_stds).add(self.t_means).mul(255).clamp(0, 255).permute(1, 2, 0)
            output_image = output_image.cpu().numpy().astype(np.uint8)
        return output_image


class WebCamera:
    def __init__(self, input_camera_device=0, virtual_camera_size=(512, 512), fps=30):
        self.video_capture = cv2.VideoCapture(input_camera_device)
        self.virtual_webcam = pyvirtualcam.Camera(width=virtual_camera_size[0], height=virtual_camera_size[1], fps=fps)

    def get_frame(self):
        return self.video_capture.read()[1]

    def send_frame(self, img):
        self.virtual_webcam.send(img)
        self.virtual_webcam.sleep_until_next_frame()

    def close(self):
        self.video_capture.release()
        self.virtual_webcam.close()


if __name__=="__main__":
    img_scale = 64 # 32
    img_size = (img_scale*16, img_scale*12) #16x9
    camera = WebCamera(virtual_camera_size=img_size)
    gan = ArcaneGAN(img_size=img_size)
    try:
        while True:
            img = camera.get_frame()
            img = gan.process(img)
            camera.send_frame(img)
    finally:
        camera.close()