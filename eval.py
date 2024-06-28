import numpy as np
import torch

# Set a manual seed for reproducibility of results
torch.manual_seed(42)

# Determine if CUDA (GPU support) is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# File name containing the ECG data
file_name = "ECG_file_example.txt"
file_features = []

# Try to load the ECG data, handle exceptions if file is missing or incorrectly formatted
try:
    fil_load_ekg = np.loadtxt(file_name)

    # Check if the data has the expected shape (5000 samples, 13 columns)
    if fil_load_ekg.shape == (5000, 13):
        # Remove the first column (assumed to be non-feature column)
        data = np.delete(fil_load_ekg, 0, axis=1)
        file_features.append(data)
    else:
        print("The data shape is not the standard 5000x13.")
except IOError:
    print("File not found or unable to read file.")
except ValueError:
    print("File content is not in the expected format.")

# Load the trained deep learning model
model_path = "model_file/model.pt"
try:
    trained_DL_Model = torch.jit.load(model_path, map_location=device)
except Exception as e:
    print(f"Error loading the model: {e}")

# Class labels
class_labels = ['AFIB', 'Flutter', 'Sinus']

# Evaluate the model
if len(file_features) > 0:
    trained_DL_Model.eval()

    for data in file_features:
        inputs = torch.Tensor(data).to(device)
        outputs = trained_DL_Model(inputs)

        # Calculate probabilities
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_indices = torch.argmax(probabilities, dim=1)
        predicted_classes = [class_labels[idx] for idx in predicted_class_indices.cpu().numpy()]
        predicted_probs = probabilities.max(dim=1)[0].cpu().detach().numpy()

        # Output results
        for idx, (cls, prob) in enumerate(zip(predicted_classes, predicted_probs)):
            print(f"File {idx + 1} is: {file_name}, Predicted Class: {cls}, Probability: {prob:.4f}")
else:
    print("No valid ECG data found for evaluation.")
