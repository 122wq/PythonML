from pathlib import Path

import torch
import torch.nn as nn


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


def export_onnx_model():
    #load the model
    model = torch.load('mnist_model.pkl', map_location="cpu", weights_only=False)
    model.eval()
    #input tensor (sample input)
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        #model to conver
        model,
        #tensor input
        dummy_input,
        'mnist_model.onnx',
        #input name for the graph
        input_names=["input"],
        output_names=["logits"],
        #make it resizable
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print("Exported ONNX model to mnist_mode.onnx")


if __name__ == "__main__":
    export_onnx_model()