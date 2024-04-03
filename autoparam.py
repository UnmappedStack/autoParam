"""
This is the main AutoParam terminal application file that gets run.
None of the actual functionality is done here, it really just connects all the other modules.
"""
from choose_params import *
from readcsv import *
from run_model import *
from train_model import *

def open_train_model():
  print("Please select a dataset file to train from. It should be a CSV file. Check the README to view the correct formatting of the file. You may get an error if it's the wrong format.")
  dataset_filename = input("Enter the dataset filename (ensure it's in the same folder as the program): ")
  model_filename = input("Choose a name for the model to be saved as once training is complete (No spaces or full stops/periods): ")
  print("Opening dataset...")
  dataset_contents = read_csv_to_array(dataset_filename)
  print("Done. Automatically selecting model parameters...")
  chosen_params = select_parameters(dataset_contents)
  print("Done. Training model with selected parameters (this may take a while depending on the size of your training data)...")
  train_model(chosen_params, dataset_contents, model_filename)
  print("Done, model has been trained and saved. Continuing to main terminal to train another model or run this model.\n")

def open_run_model():
  doRun = True
  print("Please select the name that you chose to save your model as.")
  model_filename = input("Enter the model name: ")
  try:
    print("Please enter the inputs to the model, in numerical form, each input split by a comma (,).")
    model_inputs = [float(val) for val in input("Enter the model inputs: ").split(",")]
  except:
    print("Invalid input data. Exiting to main terminal...\n")
    doRun = False
  if (doRun):
    try:
      print("Running model (this could take a while depending on the size of your training data)...")
      model_output = run_model(model_filename, model_inputs)
      print("Model has been run. Model output:\n\n", model_output[0])
      if (model_output[0] < 0.5 and model_output[0] > 0):
        print("(Most likely FALSE)")
      elif (model_output[0] >= 0.5 and model_output[0] < 1):
        print("(Most likely TRUEs)")
    except:
      print("Couldn't find a model with this name. Please try again.")
      print("Exiting to main terminal...\n")

while (True):
  print("This product's results CANNOT BE GUARANTEED and answers produced by AutoParam models MAY NOT ALWAYS BE CORRECT. The results depend on the quality of data that YOU GIVE THE MODEL during training. This product is provided WITHOUT WARRANTY, and AS-IS, EVEN if it doesn't do the specified task. See LICENSE.md for more information.")
  print("AutoParam terminal: Train neural networks without any technical knowledge\nDo you want to train a model, or run an existing model that you have saved on your computer?")
  choice = input("Type TRAIN or RUN depending on which you want to do: ").lower()
  if (choice == "train"):
    open_train_model()
  else:
    open_run_model()
