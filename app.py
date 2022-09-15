import torch
import torchvision
from torch import nn
from torchvision import transforms

from PIL import Image
from typing import List, Tuple, Dict

import streamlit as st
import os
import json
import random
import boto3
from io import BytesIO


# Create an instance of efficientnetv2_s with pretrained weights, feeze the base model layers, and change the classifier head.
def create_effnetv2_s():
  
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
    p = 0.2
    in_features = 1280
    out_features = len(label_names)

    for param in model.features.parameters():
        param.requires_grad = False
      
    model.classifier = nn.Sequential(nn.Dropout(p=p, inplace=True),
                                     nn.Linear(in_features=in_features,
                                               out_features=out_features))   
    return model


# Make predicts
def prediction(model: torchvision.models, image: Image, label_names: List[str]) -> Tuple[str, float]:

    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.TrivialAugmentWide(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std =[0.229, 0.224, 0.225])])
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    logits = model(transformed_image)
    probs = torch.softmax(logits, dim=1)
    label = torch.argmax(probs, dim=1)

    prob = round(probs.max().item(), 3)
    label_name = label_names[label]

    return label_name, prob


# Get all unseen images of each label and convert labels into Python List.
with open("data/food-101/meta/test.json", "r") as f:
    test_json_dict = json.load(f)
    images_dict = {key : test_json_dict[key] for key in test_json_dict.keys()}
    label_names = sorted(images_dict.keys())


# Load PIL images from s3 to make dictionary with keys
@st.cache
def load_files_from_s3(keys: List[str],
                       bucket_name: str = "food101-classification-bucket") -> Dict[str, Image.Image]:
    s3 = boto3.client('s3')
    s3_files_dict = {}
    for key in keys:
        s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
        s3_file_cleaned = s3_file_raw['Body'].read()
        image_from_s3 = Image.open(BytesIO(s3_file_cleaned))
        s3_files_dict[key] = image_from_s3
    
    return s3_files_dict


if __name__=='__main__':

    # Model weights trained on CUDA but CPU is default here.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.title("Welcom to Food Vision :heart:")
    instructions = """
                    **Upload** your own food image or **select** one at sidebar.\n
                    Images the model has not seen are included in selection.\n                    
                    See what Neural Network predicts.                   
                    The output will be displayed to the below.
                    """
    st.write(instructions)

    model_path = "models/Food101_EffNetV2-S_5-epochs.pth"
    model = create_effnetv2_s()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    uploaded = st.file_uploader("Upload Food Image")
    if uploaded:
        image = Image.open(uploaded)
        label_name, prob = prediction(model=model, image=image, label_names=label_names)
        caption = "Here is the image you've uploaded."
        instruction = "Click **X** the above if you want to select image."
    else:
        food_type = st.sidebar.selectbox("Food Type", label_names)
        food_name = st.sidebar.selectbox("Food Image Name", images_dict[food_type])

        keys = ["images/" + key + ".jpg" for key in images_dict[food_type]]
        selected = "images/" + food_name + ".jpg"
        s3_files_dict = load_files_from_s3(keys=keys)

        image = s3_files_dict[selected]

        label_name, prob = prediction(model=model, image=image, label_names=label_names)
        caption = f"Selected food type is '{food_type}'"
        instruction = ''

    resized_image = image.resize((384,384))
    st.image(resized_image, caption=caption)
    
    descriptions = f"""
                    Neural Network predicts your image as **{label_name}** with a probability of **{prob}**.

                    :eyes:

                    """
    st.write(descriptions, instruction)
