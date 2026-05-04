import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

def trend_plot(data, name, output_dir='./Fig'):
    """
    Generate a trend plot to visualize how predictions and labels match each other.

    Args:
        data: data set containing predictions and labels. The first column of data is ID. If you don't have ID, you can just set is as 0. The second column is pred and the third column is label. Each row is a data sample.
        name: name of the vital signal
        output_dir: directory to save the plot (default: './Fig')

    Outputs:
        Saves the trend plot as a PNG file in the specified directory.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    pred = data[:, 1].flatten()
    label = data[:, 2].flatten()

    # Sort predictions and labels based on the order of labels
    index = np.argsort(label)
    pred = pred[index]
    label = label[index]

    # Calculate metrics
    mae = np.mean(np.abs(pred - label))  # Mean Absolute Error
    me = np.mean(pred - label)          # Mean Error
    correlation_matrix = np.corrcoef(pred, label)  # Correlation coefficient matrix
    correlation = correlation_matrix[0, 1]        # Extract correlation value
    std = np.std(pred - label)          # Standard deviation of differences

    # Calculate density for scatter plot
    xy = np.vstack([label, pred])       # Stack labels and predictions for density calculation
    density = gaussian_kde(xy)(xy)     # Kernel density estimation
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B3B3EB", "#3636FF", "#00006C"])

    # Create the trend plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(np.arange(1, pred.shape[0]+1), pred, c=density, cmap=custom_cmap, s=1)  # Scatter plot of predictions
    plt.scatter(np.arange(1, pred.shape[0]+1), label, c="red", s=1)                            # Scatter plot of labels
    plt.colorbar(scatter, label='Density')  # Add color bar for density
    plt.legend(["Prediction", "Label"], loc="upper left")
    
    # Add text with metrics in the plot
    plt.text(1, 0, f"MAE:{mae:6.2f} ME:  {me:6.2f}\nSTD: {std:6.2f} Corr: {correlation:3.2f}", 
             fontsize="x-large", ha='right', va='bottom', transform=plt.gca().transAxes)
    
    # Add title and save the plot
    name_with_space = name.replace('_', ' ')
    plt.title(f'Trend Plot of {name_with_space}')
    output_path = os.path.join(output_dir, 'Trend_Plot_' + name + '.png')
    plt.savefig(output_path)
    plt.close()

def bland_altman_plot(data, name, output_dir='./Fig'):
    """
    Generate a Bland-Altman plot to visualize the difference between predictions and labels.

    Args:
        data: data set containing predictions and labels. The first column of data is ID. If you don't have ID, you can just set is as 0. The second column is pred and the third column is label. Each row is a data sample.
        name: name of the vital signal (e.g., SP_on_sample_level)
        output_dir: directory to save the plot (default: './Fig')

    Outputs:
        Saves the Bland-Altman plot as a PNG file in the specified directory.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the mean and difference between predictions and labels
    pred = data[:, 1].flatten()
    label = data[:, 2].flatten()

    mean = np.mean([pred, label], axis=0)
    diff = pred - label
    mean_diff = np.mean(diff)  # Mean of the differences
    std_diff = np.std(diff)    # Standard deviation of the differences

    # Calculate density for scatter plot
    xy = np.vstack([mean, diff])       # Stack mean and diff for density calculation
    density = gaussian_kde(xy)(xy)     # Kernel density estimation
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B3B3EB", "#3636FF", "#00006C"])

    # Create the Bland-Altman plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mean, diff, c=density, cmap=custom_cmap, s=10)  # Scatter plot with density
    plt.colorbar(scatter, label='Density')  # Add color bar for density
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean Difference ({mean_diff:.2f})')  # Mean line
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='-.', label=f'+1.96 SD ({mean_diff + 1.96 * std_diff:.2f})')  # +1.96 SD line
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label=f'-1.96 SD ({mean_diff - 1.96 * std_diff:.2f})')  # -1.96 SD line
    
    # Add labels, title, and legend
    plt.xlabel('Mean of Two Measurements')
    plt.ylabel('Difference Between Measurements')
    name_with_space = name.replace('_', ' ')
    plt.title(f'Bland-Altman Plot of {name_with_space}')
    plt.legend()
    
    # Save the plot to a file
    output_path = os.path.join(output_dir, 'Bland_Altman_Plot_' + name + '.png')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    data_file = '/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy'
    data = np.load(data_file)
    
    bland_altman_plot(data, 'SP_on_sample_level')
    plt.show()