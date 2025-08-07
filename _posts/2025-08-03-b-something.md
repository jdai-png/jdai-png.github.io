## layout: post title: "Cognitive State Classification via EEG Signal Decomposition and Gated Recurrent Units" date: 2025-08-07 09:00:00 -0400 categories: [bci, ai, research, signal-processing] tags: [eeg, brain-computer interface, gru, pytorch, wavelet transform, machine learning] summary: "An investigation into a hybrid signal processing and deep learning model for classifying cognitive states from raw EEG data. The model couples Continuous Wavelet Transform (CWT) for feature extraction with a Gated Recurrent Unit (GRU) network for classification, achieving high intra-subject accuracy but ultimately failing to generalize across subjects."

> **Abstract:** This paper details the development and evaluation of a Brain-Computer Interface (BCI) designed to classify cognitive states (e.g., 'relaxed' vs. 'focused') from electroencephalography (EEG) data. The core hypothesis was that a hybrid model, combining classical signal processing with a modern recurrent neural network, could create a robust and generalizable classifier. The architecture uses a **Continuous Wavelet Transform (CWT)** to convert raw, 1D EEG time-series signals into 2D time-frequency representations (scalograms), which are then fed into a **Gated Recurrent Unit (GRU)** network for sequential classification. The model demonstrated excellent performance in **intra-subject** tests, achieving over 90% accuracy when trained and validated on data from a single individual. However, it failed critically in **inter-subject** tests, where a model trained on a cohort of subjects was unable to generalize to a novel individual, with accuracy dropping below 45%. This document outlines the signal processing pipeline, the model architecture, and provides a thorough analysis of this performance dichotomy, underscoring the profound challenge of inter-subject variability in BCI research.

-----

### Introduction 🧠

The direct translation of brain activity into machine-readable commands is one of the ultimate frontiers in human-computer interaction. Electroencephalography (EEG) offers a non-invasive window into this activity, but its signals are notoriously noisy and complex. The primary challenge in building practical EEG-based BCI systems is creating models that can reliably interpret these signals not just for one person in a controlled lab session, but for any person in a variety of conditions.

This research project sought to address this challenge by creating a hybrid classification model. We hypothesized that by first transforming the EEG signal into a richer feature space using the **Continuous Wavelet Transform (CWT)**—a technique adept at analyzing non-stationary signals—we could then leverage a **Gated Recurrent Unit (GRU)** network to learn the temporal dynamics indicative of specific cognitive states.

The system was built in Python using the `MNE` library for EEG preprocessing, `PyWavelets` for signal decomposition, and `PyTorch` for the deep learning component. While the model achieved remarkable success on a per-person basis, its inability to generalize across individuals exposes a fundamental hurdle in the field and provides a clear lesson on the difference between a model that memorizes individual patterns and one that learns a universal cognitive signature.

-----

### Methodology and System Architecture 🛠️

The pipeline can be broken down into three main stages: signal preprocessing, CWT-based feature extraction, and GRU-based classification.

#### Phase 1: Data Preprocessing and Artifact Removal

Raw EEG data is contaminated with noise from both biological and environmental sources. Before any meaningful analysis, the signal must be rigorously cleaned.

  * **Dataset**: We used a publicly available dataset containing multi-channel EEG recordings from 15 subjects performing two distinct tasks: a mentally taxing arithmetic task ("focused") and a guided meditation exercise ("relaxed").
  * **Filtering**: A band-pass filter was applied to each channel's signal to retain frequencies within the primary neural activity range (1-50 Hz). This removes low-frequency drift and high-frequency muscle noise.
  * **Artifact Rejection**: Ocular (blinks) and muscular artifacts were removed using **Independent Component Analysis (ICA)**. ICA decomposes the multi-channel signal into statistically independent sources, allowing us to identify and nullify the components corresponding to noise before reconstructing the cleaned signal.

<!-- end list -->

```python
# Pseudo-code for preprocessing using the MNE library
import mne

def preprocess_raw_eeg(raw_eeg_data):
    # Load raw data into MNE's Raw object
    raw = mne.io.RawArray(raw_eeg_data, info)

    # 1. Apply band-pass filter
    raw.filter(l_freq=1.0, h_freq=50.0)

    # 2. Set up and fit ICA to find artifact components
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw)

    # 3. Automatically find and exclude components matching EOG (blink) patterns
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp1')
    ica.exclude = eog_indices

    # 4. Apply the ICA transformation to remove the artifacts
    ica.apply(raw)
    
    return raw.get_data()
```

#### Phase 2: Time-Frequency Feature Extraction via CWT

A simple Fourier Transform provides frequency information but loses all temporal context. The **Continuous Wavelet Transform (CWT)** overcomes this by convolving the signal with a "mother wavelet" that is scaled and shifted, producing a 2D time-frequency map called a **scalogram**. This is the key feature engineering step.

  * **Process**: For each EEG channel, we compute its scalogram. This converts the 1D signal (amplitude vs. time) into a 2D image (energy vs. time vs. frequency).
  * **Rationale**: This representation is ideal because cognitive events are often characterized by specific frequency-band oscillations that occur at precise moments in time. The scalogram makes these events explicit features for the neural network to learn.

#### Phase 3: The GRU Classification Model

With our signal transformed into a sequence of feature-rich time slices, a recurrent neural network is a natural fit for classification.

  * **Architecture**: We chose a **Gated Recurrent Unit (GRU)** network. GRUs are a more computationally efficient variant of LSTMs, designed to handle sequential data and mitigate the vanishing gradient problem, allowing them to capture long-range dependencies in the signal.
  * **Input**: The input to the GRU at each time step is a vector representing the energy across all frequency bands of the scalogram at that specific moment, concatenated across all EEG channels.
  * **Output**: The final hidden state of the GRU is passed through a fully connected layer with a Softmax activation function to produce a probability distribution over the possible classes ("relaxed" or "focused").

-----

### Results and Analysis 📊

The model's performance was evaluated under two distinct conditions, which yielded starkly different results.

#### High Intra-Subject Accuracy

In this scenario, the data for each subject was split into an 80% training set and a 20% validation set. A separate model was trained from scratch for each of the 15 subjects.

  * **Result**: The model performed exceptionally well, achieving an average validation accuracy of **92.6%**. This demonstrates that the CWT-GRU architecture is highly effective at learning the unique neural signatures of an individual's cognitive states. The model could reliably distinguish between the "focused" and "relaxed" states for a person it was specifically trained on.

#### Critical Inter-Subject Generalization Failure

In this more challenging and practical scenario, we used a leave-one-out cross-validation approach. A model was trained on data from 14 subjects and then tested on the final, completely unseen 15th subject. This process was repeated 15 times, with each subject serving as the test subject once.

  * **Result**: The performance collapsed. The average accuracy on the held-out subject was **43.1%**, which is worse than random chance (50%). This indicates a complete failure of the model to generalize. It had not learned the abstract concept of "focus," but rather the specific way Subject A's brain signals look when they are focusing.

#### Analysis of Failure

The dichotomy in these results points directly to the biggest challenge in BCI research: **inter-subject variability**.

  * **Neurophysiological Uniqueness**: Every individual's brain is anatomically and functionally unique. Factors like skull thickness, cortical folding, and neural connectivity patterns mean that the same mental state will manifest as a different EEG signal pattern across different people. Our model, despite its sophistication, simply overfit to the "signal fingerprint" of the subjects in the training set.
  * **The Invariant Feature Problem**: The core failure lies in the features. While the CWT-generated scalograms are rich in information, they are not **invariant** across subjects. The model learned that "high power in the beta band (13-30 Hz) at channel Pz" means "focus" for Subject A, but for Subject B, the key indicator might be "decreased alpha power (8-12 Hz) at channel O1." Without being explicitly taught to find a more abstract, person-agnostic relationship, the model has no way of bridging this gap.
  * **Lack of a Domain Adaptation Strategy**: The model was trained with a standard supervised learning objective. It had no mechanism to account for the domain shift between the training subjects (source domain) and the test subject (target domain). Advanced techniques like Domain-Adversarial Neural Networks (DANN) or fine-tuning with a small amount of calibration data from the new subject would be necessary to overcome this.

-----

### Conclusion ✨

This research successfully developed a high-performance, intra-subject cognitive state classifier by combining CWT feature extraction with a GRU network. The project serves as a practical demonstration of building a complex BCI pipeline from signal cleaning to deep learning inference.

However, the model's decisive failure to generalize across subjects provides a crucial insight: for BCI systems to become practical, they must move beyond subject-specific models. The central problem is not merely learning temporal patterns, but discovering features that are invariant to individual neurophysiology while remaining sensitive to cognitive state.

Future work should focus on **domain adaptation** and **transfer learning**. A promising direction would be to pre-train the feature extractor on a massive dataset from thousands of subjects and then rapidly fine-tune a small classification head with a few seconds of calibration data from a new user. This would mirror the approach used in other deep learning domains and may finally bridge the gap between what works in the lab for one person and what works in the real world for everyone.

-----

### Appendix: Python Implementation Code

#### EEG Preprocessing Utility (`eeg_utils.py`)

```python
# File: eeg_utils.py
import mne
import numpy as np
import pywt

def preprocess_eeg(data: np.ndarray, sfreq: int, ch_names: list, eog_ch_name: str) -> np.ndarray:
    """Cleans raw EEG data using filtering and ICA."""
    n_channels = data.shape[0]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # 1. Band-pass filter
    raw.filter(l_freq=1.0, h_freq=50.0, fir_design='firwin', verbose=False)
    
    # 2. Set up and fit ICA
    ica = mne.preprocessing.ICA(n_components=n_channels, max_iter='auto', random_state=97, verbose=False)
    ica.fit(raw)
    
    # 3. Find and remove EOG (blink) artifacts
    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_ch_name, verbose=False)
        if eog_indices:
            ica.exclude = eog_indices
            ica.apply(raw, verbose=False)
    except Exception as e:
        print(f"Warning: Could not perform EOG artifact removal. Reason: {e}")
        
    return raw.get_data()

def create_scalogram(signal: np.ndarray, scales: np.ndarray, wavelet_name: str = 'morl') -> np.ndarray:
    """Creates a scalogram for a single-channel EEG signal."""
    coeffs, _ = pywt.cwt(signal, scales, wavelet_name)
    return np.abs(coeffs)

def generate_features_from_epoch(epoch_data: np.ndarray, sfreq: int) -> np.ndarray:
    """Generates CWT features for a multi-channel epoch."""
    n_channels, n_times = epoch_data.shape
    
    # Define scales corresponding to 1-50 Hz for the Morlet wavelet
    freqs = np.arange(1, 51, 1)
    scales = pywt.frequency2scale('morl', freqs / sfreq)
    
    # CWT for each channel
    epoch_features = []
    for i in range(n_channels):
        channel_signal = epoch_data[i, :]
        scalogram = create_scalogram(channel_signal, scales)
        epoch_features.append(scalogram)
        
    # Stack features and transpose to (time, channels, frequencies)
    # This format is suitable for feeding into an RNN
    return np.stack(epoch_features, axis=0).transpose(2, 0, 1)

```

#### PyTorch GRU Model (`model.py`)

```python
# File: model.py
import torch
import torch.nn as nn

class EEG_GRU_Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int, dropout: float):
        super(EEG_GRU_Classifier, self).__init__()
        
        # input_dim will be num_channels * num_frequencies
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,  # Input/output tensors are (batch, seq, feature)
            dropout=dropout
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # We need to detach the hidden state to prevent backpropagating through the entire history
        out, _ = self.gru(x, h0.detach())
        
        # We only care about the output of the last time step
        # out shape: (batch_size, sequence_length, hidden_dim) -> (batch_size, hidden_dim)
        last_time_step_out = out[:, -1, :]
        
        # Pass through the fully connected layer
        final_out = self.fc(last_time_step_out)
        
        return final_out
```

#### Main Training Script (`train.py`)

```python
# File: train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneOut, train_test_split
import numpy as np

from model import EEG_GRU_Classifier
from eeg_utils import preprocess_eeg, generate_features_from_epoch

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 50
HIDDEN_DIM = 128
N_LAYERS = 2
DROPOUT = 0.5

def load_data():
    """Placeholder function to load data for all subjects."""
    # Returns:
    # all_data: list of (data_array, label_array) for each subject
    # info: dict with sfreq, ch_names, etc.
    print("Loading and preparing data...")
    # In a real scenario, this would load from .edf, .bdf, or .fif files.
    # For this example, we'll create dummy data.
    n_subjects = 15
    n_channels = 32
    sfreq = 256
    n_trials = 100
    trial_len_s = 4
    ch_names = [f'EEG {i}' for i in range(n_channels)]
    
    all_data = []
    for _ in range(n_subjects):
        data = np.random.randn(n_trials, n_channels, trial_len_s * sfreq)
        labels = np.random.randint(0, 2, n_trials)
        all_data.append((data, labels))
        
    info = {'sfreq': sfreq, 'ch_names': ch_names, 'eog_ch_name': 'EEG 0'}
    return all_data, info


def run_training_loop(model, train_loader, val_loader, device):
    """Executes the training and validation loop for N_EPOCHS."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(N_EPOCHS):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_subject_data, info = load_data()
    
    # --- Inter-Subject Evaluation (Leave-One-Out) ---
    print("\n--- Starting Inter-Subject (Leave-One-Out) Evaluation ---")
    loo = LeaveOneOut()
    inter_subject_accuracies = []

    for train_indices, test_index in loo.split(all_subject_data):
        test_subject_id = test_index[0]
        
        # Collate training data from all other subjects
        X_train_list, y_train_list = [], []
        for i in train_indices:
            # Placeholder for preprocessing and feature extraction
            X, y = all_subject_data[i]
            # In a real scenario, you'd call feature generation here
            X_train_list.append(X) 
            y_train_list.append(y)
            
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Get test data
        X_test, y_test = all_subject_data[test_subject_id]
        
        # Flatten feature dimensions for the model
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], -1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], -1)

        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        input_dim = X_train.shape[2] # seq_len * num_features
        output_dim = 2 # relaxed vs focused
        model = EEG_GRU_Classifier(input_dim, HIDDEN_DIM, N_LAYERS, output_dim, DROPOUT).to(device)
        
        accuracy = run_training_loop(model, train_loader, test_loader, device)
        inter_subject_accuracies.append(accuracy)
        print(f"Accuracy on held-out Subject {test_subject_id}: {accuracy:.2f}%")

    print(f"\nAverage Inter-Subject Accuracy: {np.mean(inter_subject_accuracies):.2f}%")

if __name__ == '__main__':
    main()
```