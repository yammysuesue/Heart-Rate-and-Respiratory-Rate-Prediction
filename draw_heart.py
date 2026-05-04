import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

data = np.load("/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy")


# Filter the data
b, a = butter(2, [0.5/50, 5/50], btype='band')  # fs=100Hz → Nyquist=50Hz


# compare the original data and the filtered data, in one figure
original_data = data[0, :1000]
filtered_data = filtfilt(b, a, original_data)

# draw the original data and the filtered data, in one figure
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].plot(original_data)
axes[0].set_title('Original Data')
axes[1].plot(filtered_data)
axes[1].set_title('Filtered Data')
plt.show()

