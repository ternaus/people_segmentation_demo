"""Streamlit web app"""

import numpy as np
import streamlit as st
from PIL import Image
from people_segmentation.pre_trained_models import create_model
import albumentations as albu
import torch
import cv2
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache(allow_output_mutation=True)
def cached_model():
    model = create_model("Unet_2020-07-20")
    model.eval()
    return model


model = cached_model()
transform = albu.Compose([albu.Normalize(p=1)], p=1)

st.title("Segment people")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting faces...")

    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    dst = cv2.addWeighted(
        image,
        1,
        (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8),
        0.5,
        0,
    )

    st.image(mask * 255, caption="Mask", use_column_width=True)
    st.image(dst, caption="Image + mask", use_column_width=True)
