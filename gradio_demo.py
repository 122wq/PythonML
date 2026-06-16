from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


MODEL_PATH = Path(__file__).with_name("mnist_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

model = CNN()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def load_example_digits(limit=5):
    dataset = datasets.MNIST(root="./data", train=False, download=False)
    examples = []
    for index in range(min(limit, len(dataset))):
        image, _ = dataset[index]
        examples.append(ImageOps.invert(image.convert("L")))
    return examples


def extract_image_payload(image):
    if isinstance(image, dict):
        return image.get("composite") or image.get("image") or image.get("background")
    return image


def predict_digit(image: Image.Image):
    image = extract_image_payload(image)
    if image is None:
        return {}

    image = ImageOps.autocontrast(image.convert("L"))
    image = ImageOps.invert(image)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    return {str(index): float(probability) for index, probability in enumerate(probabilities)}


demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        label="Draw a digit",
        type="pil",
    ),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="MNIST Digit Classifier",
    description="Upload or draw a handwritten digit and the app will classify it using mnist_model.pth.",
    examples=load_example_digits(),
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch()
