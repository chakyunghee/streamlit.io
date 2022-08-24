import torch
import torchvision
from torch import nn
from torchvision import transforms

from PIL import Image
from typing import List, Tuple

import streamlit as st


# Create an instance of efficientnetv2_s with pretrained weights, feeze the base model layers, and change the classifier head.
def create_effnetv2_s():
  
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=weights)
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
    logits = model(transformed_image)
    probs = torch.softmax(logits, dim=1)
    label = torch.argmax(probs, dim=1)

    prob = round(probs.max().item(), 3)
    label_name = label_names[label]

    return label_name, prob


# Convert labels in a text file into Python List.
with open("labels.txt", "r") as f:
    labels = f.read()
    label_names = labels.split("\n")
    label_names.remove('')


if __name__=='__main__':

    st.title("Welcom to Food Vision!")
    instructions = """
                    Upload your own food image to see what Neural Network predicts.
                    The output will be displayed to the below.
                    """
    st.write(instructions)

    model_path = "models/Food101_EffNetV2-S_5-epochs.pth"
    model = create_effnetv2_s()
    model.load_state_dict(torch.load(model_path))

    uploaded = st.file_uploader("Upload Food Image")
    if uploaded:
        image = Image.open(uploaded)
        label_name, prob = prediction(model=model, image=image, label_names=label_names)

        st.title("Here is the image you've uploaded.")
        resized_image = image.resize((224,224))
        st.image(resized_image)
        descriptions = f"""
                        Neural Network predicts your image as {label_name} with a probability of {prob}.
                        """
        st.write(descriptions)
        st.title("Try again if you'd like to.")


