#
import streamlit as st
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler

from PIL import Image
import os
import os.path
import sys
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


st.title("Multiclass image classification with ResNet50")
st.write('This is a web app to classify in which sporting goods retailers belongs an image.')
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    us_image = np.array(Image.open(img_file_buffer))
else:
    demo_image = "bruno-nascimento-unsplash.jpg"
    us_image = np.array(Image.open(demo_image))


# LOAD THE ENTIRE MODEL
#MobilenetV2
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location=torch.device('cpu')

sporting_model= torch.load('classif_all_resnet50_sporting.pt',map_location=map_location)

#the directory containing the training data
root_dir='sporting_goods_retailer\\'


#PREPROCESS THE INPUT IMAGE
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

sporting_dataset = datasets.ImageFolder(root = root_dir + "train",
                                      transform = image_transforms["train"]
                                     )

#sporting_dataset.class_to_idx
idx2class = {v: k for k, v in sporting_dataset.class_to_idx.items()}

#INFERENCE
def pil_loader(path):    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def makePrediction(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = image_transforms["test"]

    #test_image= pil_loader(image)    
    test_image = Image.fromarray(np.uint8(image)).convert('RGB')

    test_image = Image.fromarray(image.astype('uint8'), 'RGB')

    plt.imshow(test_image)

    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)       
    test_image_tensor = test_image_tensor.view(3, 224, 224).to(device)    
    
    with torch.no_grad():
        model.eval()
        test_image_tensor = test_image_tensor.unsqueeze(0)        
        out = model(test_image_tensor)        
        sm=torch.nn.Softmax(dim=1)
        proba=sm(out)        
        topk, topclass = proba.topk(1, dim=1)       
              
        numpy_topclass=topclass.detach().cpu().numpy()        
        pred_proba=numpy_topclass[0][0]
        res_class="class : "+idx2class[pred_proba]             
        numpy_topk=topk.detach().cpu().numpy()        
        pred_proba=numpy_topk[0][0]
        res_proba="probability : "+str(pred_proba)         
        return res_class, res_proba

res=makePrediction(sporting_model, us_image)


st.image(
    us_image, caption=f"Photo by Bruno Nascimento on Unsplash", use_column_width=True,
)
if st.button('Predict'):
	res