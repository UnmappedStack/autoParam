# This module runs an existing model.

import torch.nn as nn
import torch
import pickle

class CustomModel(nn.Module):
    def __init__(self, num_inputs, num_hidden_layers, nodes_per_hidden_layer, output_type):
        super(CustomModel, self).__init__()
        self.input_layer = nn.Linear(num_inputs, nodes_per_hidden_layer)
        self.hidden_layers = nn.ModuleList([nn.Linear(nodes_per_hidden_layer, nodes_per_hidden_layer) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(nodes_per_hidden_layer, 1)
        self.activation = nn.ReLU() if output_type == "relu" else nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x

def run_model(model_filename, input_data):
    # Load the model
    num_inputs = len(input_data)
    picklefilename = model_filename + ".pickle"
    with open(picklefilename, 'rb') as handle:
      model_params = pickle.load(handle)
    num_hidden_layers = model_params["num_of_layers"]
    output_type = model_params["output_type"]
    nodes_per_hidden_layer = model_params["nodes_per_layer"]
    model = CustomModel(num_inputs, num_hidden_layers, nodes_per_hidden_layer, output_type)
    model.load_state_dict(torch.load((model_filename + ".pt")))
    model.eval()
    # Make predictions
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        predictions = model(input_tensor)
    return predictions.numpy()  # Convert predictions to numpy array
