# This module is a core part of the program. It selects the best hyperparameters for the module based on the training data.
import math

def select_parameters(dataset):
  # First, select the number of epochs & learning rate
  # This should be done based on the number of training examples
  num_examples = len(dataset)
  epoch_options = [100, 50, 30, 10, 3]
  lr_options = [0.001, 0.003, 0.005, 0.007, 0.009]
  eg_maxes = [15, 50, 500, 2000]
  epochs = 0
  learning_rate = 0
  for max in range(len(eg_maxes)):
    if (num_examples < eg_maxes[max]):
      epochs = epoch_options[max] - 3
      learning_rate = lr_options[max] - 0.009
  learning_rate += 0.009
  epochs += 3 # 3 is taken away above then added in the case that there are more examples than 2000, in which case it sets it to 3. Same with the line above.
  # Now work out the model size, starting with the number of layers
  # This is based on the number of features
  num_of_features = len(dataset[0])
  # A larger number of features will require a larger amount of layers. In this case, it'll linearly correspond, minus 1 layer
  num_of_layers = num_of_features - 2
  # Now calculate the best number of nodes per layer. To keep it simple, every layer will have the same number of nodes, rather than a triangle structure.
  nodes_per_layer = round(math.sqrt((num_of_layers - 1) / 1))
  # Now select if it will produce a linear output or a logistic output.
  # Create an array of all the outputs
  outputs = [dataset[i][-1] for i in range(len(dataset))]
  # If any of them are a number that isn't 1 or 0, select linear, otherwise, logistic.
  # To save time, just check the first hundred examples (or max examples if less than 100)
  output_type = "sigmoid"
  for output in outputs[:min(100,num_examples - 1)]:
    if output not in (1, 0):
      # Change to linear and break
      output_type = "linear"
      break
  return {"epochs": epochs, "learning_rate": learning_rate, "num_of_layers": num_of_layers, "output_type": output_type, "nodes_per_layer": nodes_per_layer}
