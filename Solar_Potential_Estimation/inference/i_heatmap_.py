import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import os

# Load your data
main_dir = 'test_crowd/original'
roof_label_path = f'{main_dir}/com_bin'
roof_data_path = f'{main_dir}/roof_data'

for file in os.listdir(roof_data_path):

    # print(file)
    filename = os.path.splitext(file)[0]
    # print(filename)

    data_path = os.path.join(roof_data_path, filename + '.csv')
    data = pd.read_csv(data_path)  # Replace with your CSV file path

    # Create a dictionary mapping Roof_ID to solar potential
    roof_potential = dict(zip(data['Roof_ID'], data['Solar_potential_per_year']))

    # Load your labeled roof image (PNG with connected components)
    # labeled_img = Image.open('roof_labels.png')  # Replace with your PNG path
    # labeled_array = np.array(labeled_img)
    roof_data = os.path.join(roof_label_path, filename + '.npy')
    labeled_array = np.load(roof_data)

    # Create an empty array for the heatmap
    heatmap = np.zeros_like(labeled_array, dtype=float)

    # Map solar potential to each roof
    for roof_id, potential in roof_potential.items():
        # Assuming the pixel values in labeled_array correspond to Roof_ID
        mask = labeled_array == roof_id
        heatmap[mask] = potential

    # print(np.unique(heatmap))

    # Normalize for visualization
    normalized_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # print(np.unique(normalized_heatmap))
    # Create a custom colormap (yellow to red)
    cmap = LinearSegmentedColormap.from_list('solar', ['yellow', 'red'])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_heatmap, cmap=cmap)
    plt.colorbar(label='Solar Potential (normalized)')

    # Add title and adjust layout
    plt.title('Solar Potential per Roof (Yearly)')
    plt.axis('off')  # Turn off axis if you don't need coordinates
    plt.tight_layout()
    plt.savefig(f'{main_dir}/heatmap/{filename}.png', dpi=300, bbox_inches='tight')
    # plt.show()