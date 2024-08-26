from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import pytesseract
import openai
from openai import OpenAI
import requests
import time
import streamlit_scrollable_textbox as stx
import json

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# key = os.getenv("OPEN_API_KEY")
# client = OpenAI(api_key=key)

# Function to extract text from image
def extract_text_from_image(image):
    extracted = pytesseract.image_to_string(image)
    cleaned_text = extracted.replace('©', '')
    return cleaned_text



def granite_summarization(extracted_text):
    

   
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    
    body = {
        "input": f"""I have a text-based medical report. Please summarize the key information, including patient details, diagnosis, prescribed medications, and any recommendations. 
        Focus on extracting and condensing the most important medical information while maintaining accuracy and clarity in a paragraph.
       
        {extracted_text}   
        
        Output:""",
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
    
   
    
    access_token = os.getenv("ACCESS_TOKEN")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.post(
        url,
        headers=headers,
        json=body
    )

    print(response)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    
    print(data)
    
    generated_text = data['results'][0]['generated_text']
    
    if generated_text:
        st.text_area("Summary",generated_text)
    else:
        st.text(data.get("output", "No summary available"))


# Load the logo image
logo1 = Image.open("icons/pulse.png")

# Display the logo image
st.image(logo1, use_column_width=False, width=60)  # Adjust width as needed
st.sidebar.title("DocSnap")
    

# Set up the Streamlit app
st.header("Medical Report")

# File uploader for image input
uploaded_file = st.sidebar.file_uploader("Choose a medical report image", type=["png", "jpg", "jpeg"])

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
    granite_summarization(extracted_text)
