from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import pytesseract
import openai
from openai import OpenAI
import requests

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# OpenAI API key
# key = os.getenv("OPEN_API_KEY")

# client = OpenAI(api_key=key)

# Set up the Streamlit app
st.header("DocSnap - Medical Report Text Extraction")

# File uploader for image input
uploaded_file = st.sidebar.file_uploader("Choose a medical report image", type=["png", "jpg", "jpeg"])

# Function to extract text from image
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# If an image is uploaded
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the image in the app
    st.image(image, caption="Uploaded Medical Report", use_column_width=True)
    
    # Extract text from the image using OCR
    extracted_text = extract_text_from_image(image)
    
    # Display the extracted text
    st.subheader("DocSnap of Your Report")
    
# Granite is taking control of the extracted data

    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    
    body = {
        "input": f"Summarize the following medical report in a concise manner, focusing on the patient's chief complaint, history of present illness, key examination findings, lab results, diagnosis, treatment plan, and any important patient education points. Ensure the summary is clear and easy to understand for both medical professionals and patients.\n\n{extracted_text}\n\nOutput:",
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200,
            "repetition_penalty": 1
        },
        "model_id": "ibm/granite-13b-chat-v2",
        "project_id": "0870e270-9bed-4a63-8b97-978cf72002ee",
        "moderations": {
            "hap": {
                "input": {
                    "enabled": True,
                    "threshold": 0.5,
                    "mask": {
                        "remove_entity_value": True
                    }
                },
                "output": {
                    "enabled": True,
                    "threshold": 0.5,
                    "mask": {
                        "remove_entity_value": True
                    }
                }
            }
        }
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_ACCESS_TOKEN"
    }

    response = requests.post(
        url,
        headers=headers,
        json=body
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    
    # Display the summarized output
    st.subheader("Summary of Your Report")
    st.text(data.get("output", "No summary available"))
    
    


