import os
import re

from models.architecture.Transformer import TransformerTimeSeries

# Path to the directory containing the models
model_directory = 'models/transformer'

def run_model(model, dataset):
    return

def main():
    # Loop through each file in the directory
    for filename in os.listdir(model_directory):
        # Check if the filename matches the expected pattern
        if filename.endswith(".pth") and "transformer" in filename:
            # Extract the number of layers and d_model from the filename
            match = re.search(r"transformer_(\d+)l_(\d+)d\.pth", filename)
            if match:
                num_layers = int(match.group(1))
                d_model = int(match.group(2))
                
                # Print the file's path for debugging
                file_path = os.path.join(model_directory, filename)
                print(f"File: {file_path}")
                
                # Print the extracted parameters
                print(f"Number of layers: {num_layers}")
                print(f"d_model: {d_model}")
                
                # Example of reading the file's contents - Adjust according to your file handling needs
                # with open(file_path, 'rb') as file:
                #     content = file.read()
                #     print(content)
            else:
                print(f"Filename does not match expected pattern: {filename}")
        else:
            print(f"Skipped non-matching file: {filename}")

if __name__ == '__main__':
    main()