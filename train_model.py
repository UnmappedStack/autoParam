# This module uses PyTorch to actually train the model, using the given training data and the hyperparameters chosen in the `choose_params.py` module.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
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

def train_model(parameters, training_data, model_name):
    num_inputs = len(training_data[0]) - 1
    model = CustomModel(num_inputs, parameters["num_of_layers"], parameters["nodes_per_layer"], parameters["output_type"])
    if parameters["output_type"] == "sigmoid":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    X_train = torch.tensor([[float(part) for part in example[:-1]] for example in training_data], dtype=torch.float32)
    y_train = torch.tensor([float(example[-1]) for example in training_data], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=math.floor(len(X_train) / 2), shuffle=True)
    for epoch in range(parameters["epochs"]):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model_name.replace(".", "") + ".pt")
    # Save the parameters about the model with pickle (Pickles are yum)
    picklefilename = model_name + ".pickle"
    with open(picklefilename, 'wb') as handle:
      pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
