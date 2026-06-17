from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from torchvision import datasets, transforms
MODEL_PATH = Path(__file__).with_name("mnist_model.onnx")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

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
    tensor = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
    outputs = session.run(["logits"], {"input": tensor})

    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    return {str(index): float(probability) for index, probability in enumerate(probabilities)}


demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        label="Draw a digit",
        type="pil",
    ),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="MNIST Digit Classifier",
    description="Upload or draw a handwritten digit and the app will classify it using mnist_model.onnx.",
    examples=load_example_digits(),
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch()
