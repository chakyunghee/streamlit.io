import torch
import torchvision
from torch import nn

from typing import List, Tuple
from PIL import Image
from io import BytesIO

import streamlit as st
import json
import boto3


# Create an instance of efficientnetv2_s with pretrained weights, feeze the base model layers, and change the classifier head.
def create_vit_b_16_swag():
  
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = torchvision.models.vit_b_16(weights=weights).to(device)

    for param in model.parameters():
        param.requires_grad = False
      
    model.heads = nn.Sequential(nn.Linear(in_features=768,
                                          out_features=len(label_names)))
    return model


# Make predicts
def prediction(model: torchvision.models, image: Image, label_names: List[str]) -> Tuple[str, float]:

    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    transform = weights.transforms()

    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    logits = model(transformed_image)
    probs = torch.softmax(logits, dim=1)
    label = torch.argmax(probs, dim=1)

    prob = round(probs.max().item(), 3)
    label_name = label_names[label]

    return label_name, prob


# Get all unseen images of each label and convert labels into Python List.
with open("test.json", "r") as f:
    test_json_dict = json.load(f)
    images_dict = {key : test_json_dict[key] for key in test_json_dict.keys()}
    label_names = sorted(images_dict.keys())


# Load PIL images from s3 to make dictionary with keys
def load_image_from_s3(key: str,
                       bucket_name: str = "food101-classification-bucket") -> Image.Image:
                       
    s3 = boto3.client('s3')
    s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
    s3_file_cleaned = s3_file_raw['Body'].read()
    image = Image.open(BytesIO(s3_file_cleaned))
    
    return image


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

    model_path = "models/Food101_ViT-B-16-SWAG_10-epochs.pth"
    model = create_vit_b_16_swag()
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

        key = "images/" + food_name + ".jpg"
        image = load_image_from_s3(key=key)
    
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
