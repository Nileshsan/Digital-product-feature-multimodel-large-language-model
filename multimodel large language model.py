# install openai, transformers, torch, clip-api-service


import openai
import torch
import clip
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load GPT model for text generation
gpt_model = GPT2LMHeadModel.from_pretrained('gpt4')
tokenizer = GPT2Tokenizer.from_pretrained('gpt4')
gpt_model = gpt_model.to(device)

# Preprocess the image (input object, screenshot)
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    return image

# Generate CLIP embedding

def get_clip_embedding(image, context=""):


    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Tokenize the context (if provided)
    if context:
        text_tokens = clip.tokenize([context]).to(device)
    else:
        text_tokens = clip.tokenize(["This is an image"]).to(device)

    # Generate image and text embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(text_tokens)

    return image_features, text_features

# Generate test case using GPT
def generate_test_cases(text_prompt):
    inputs = tokenizer.encode(text_prompt, return_tensors="pt").to(device)
    outputs = gpt_model.generate(inputs, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function to detect features and generate test cases
def detect_features_and_generate_tests(image_path, context=""):
    # Get embeddings from CLIP
    image_features, text_features = get_clip_embedding(image_path, context)

    # Combine the image and text features (you can apply more complex processing here)
    combined_features = image_features + text_features

    # Generate text description from features
    # This is a placeholder; you would use the combined features with a custom head to generate descriptions
    description = "Detected digital features in the image related to " + context

    # Generate test cases based on the description
    test_cases = generate_test_cases(f"Test cases for: {description}")

    return test_cases

def generate_test_cases(description):
    prompt = f"""Generate test cases for the following digital feature description:
    Feature: {description}
    1. Pre-condition:
    2. Testing steps:
    3. Expected results:

    """
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = gpt_model.generate(inputs, max_length=250, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example custom classifier head
import torch.nn as nn

class FeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Add custom head to CLIP's image features
custom_head = FeatureClassifier(input_dim=512, num_classes=10).to(device)


# Example usage
image_path = "//content//redbus2.png"
context = "Login screen for web application"


test_cases = detect_features_and_generate_tests(image_path, context)


print(test_cases)
