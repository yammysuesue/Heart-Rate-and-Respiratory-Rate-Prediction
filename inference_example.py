"""
Simple example script for using the trained model for inference
"""
import torch
import numpy as np
from ppg_hr_rr_model import HRRRPredictor, filter_ppg_signal, normalize_signals


def predict_hr_rr(model_path, signal, apply_filter=True, sampling_rate=100, 
                   lowcut=0.5, highcut=8.0, order=4):
    """
    Predict HR and RR from a single PPG signal
    
    Args:
        model_path: Path to the trained model weights (.pth file)
        signal: PPG signal array of shape (1000,) - 10 seconds at 100Hz
        apply_filter: Whether to apply bandpass filter (default: True)
        sampling_rate: Sampling rate in Hz (default: 100)
        lowcut: Low cutoff frequency in Hz (default: 0.5)
        highcut: High cutoff frequency in Hz (default: 8.0)
        order: Filter order (default: 4)
    
    Returns:
        hr: Predicted heart rate (bpm)
        rr: Predicted respiratory rate (bpm)
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRRRPredictor(input_length=1000).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Preprocess signal (filter and normalize)
    signal = np.array(signal)
    
    # Apply filter if enabled
    if apply_filter:
        signal = filter_ppg_signal(
            signal, 
            sampling_rate=sampling_rate,
            lowcut=lowcut,
            highcut=highcut,
            order=order
        )
    
    # Normalize signal
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val > 0:
        signal_normalized = (signal - min_val) / (max_val - min_val)
    else:
        signal_normalized = signal
    
    # Convert to tensor and add batch dimension
    signal_tensor = torch.FloatTensor(signal_normalized).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(signal_tensor)
        hr = output[0, 0].item()
        rr = output[0, 1].item()
    
    return hr, rr


if __name__ == "__main__":
    # Example usage
    # Load a sample signal from your data
    data = np.load("/home/ghosn/Project/csee8300_3/data/dataset_constant_ibi_constant_wa.npy")
    sample_signal = data[0, :1000]  # First 1000 points (10 seconds of PPG)
    true_hr = data[0, -5]
    true_rr = data[0, -4]
    
    print(f"Sample signal shape: {sample_signal.shape}")
    print(f"True HR: {true_hr:.2f} bpm")
    print(f"True RR: {true_rr:.2f} bpm")
    
    # Predict (assuming model is trained and saved)
    try:
        hr_pred, rr_pred = predict_hr_rr('best_model.pth', sample_signal)
        print(f"\nPredicted HR: {hr_pred:.2f} bpm")
        print(f"Predicted RR: {rr_pred:.2f} bpm")
        print(f"\nHR Error: {abs(hr_pred - true_hr):.2f} bpm")
        print(f"RR Error: {abs(rr_pred - true_rr):.2f} bpm")
    except FileNotFoundError:
        print("\nModel file not found. Please train the model first using ppg_hr_rr_model.py")

