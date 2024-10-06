import os
import numpy as np
import torch
from gangsta_car.models import BiLSTMEventDetector, ConvNETEventClassifier


class PickerModel:
    def __init__(self, model_path, model_name, device):
        """
        model_path (str): Path to the directory containing the LSTM model.
        model_name (str): Filename of the model.
        device (str or torch.device): Device to load the model on.
        """
        self.device = torch.device(device)
        self.model = BiLSTMEventDetector(input_channels=1).to(self.device)

        state_dict = torch.load(
            os.path.join(model_path, model_name),
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set model to evaluation mode

    def predict(self, inputs):
        """
        inputs (torch.Tensor): One-dimensional input tensor.

        """
        #Move inputs to the correct device
        inputs = inputs.to(self.device)

        with torch.no_grad():
            output = self.model(inputs)

        # Get the index of the most propable sample
        output_index = torch.argmax(output, dim=1).cpu().numpy()

        # If batch size is 1, return the index. Otherwise, return the whole batch
        return int(output_index[0]) if output_index.size == 1 else output_index


class DetectorModel:
    def __init__(self, model_path, model_name, device, prediction_threshold=0.8):
        """
        model_path (str): Path to the directory containing CONVOLUTIONAL the model.
        model_name (str): Filename of the model.
        device (str or torch.device): Device to load the model on.
        """
        self.device = torch.device(device)
        self.model = ConvNETEventClassifier(input_channels=1).to(self.device)
        self.prediction_threshold = prediction_threshold

        state_dict = torch.load(
            os.path.join(model_path, model_name),
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set model to evaluation mode

    def predict(self, inputs):

        inputs = inputs.to(self.device)

        with torch.no_grad():

            output = self.model(inputs)

        # Apply a threshold to classify as either 0 or 1
        prediction = (output > self.prediction_threshold).float().round().item()

        return bool(prediction)