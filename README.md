# Face_AntiSpoofing
F0llowing is the Code we have used for my Project.
# Deep Ensemble Learning with Frame Skipping – MATLAB Implementation

This repository contains MATLAB code for a deep learning project focused on **video-based face anti-spoofing**, using frame skipping and ensemble learning strategies. The methods are evaluated on two benchmark datasets: **MSU-MFSD** and **CASIA-MFSD**.

## 🔍 Project Overview

This work explores two key techniques to improve spoof detection in video-based face authentication systems:
- **Frame Skipping:** Reduces redundant temporal data by selecting frames at regular intervals, lowering computational cost.
- **Ensemble Learning:** Combines outputs from multiple models (e.g., LSTM, BiLSTM, GRU) to improve classification accuracy and robustness.

## 📂 File Structure

| File Name             | Description |
|-----------------------|-------------|
| `frame_skipping.m`    | Selects frames from videos using a fixed interval (e.g., every 5th frame), helping reduce redundancy in video analysis. |
| `ensemble_learning.m` | Implements ensemble decision fusion across deep learning models to enhance final prediction performance. |

## 📊 Datasets Used

This project was evaluated on the following publicly available datasets:
- **MSU-MFSD**: A benchmark dataset for face anti-spoofing, containing real and spoofed face videos.
- **CASIA-MFSD**: Another widely used dataset with diverse attack types (e.g., print, replay) across multiple devices.

Please refer to the official dataset providers for access and licensing terms.

## 🛠 Requirements

- MATLAB R2020a or later
- Deep Learning Toolbox
- Image Processing Toolbox (for video/frame processing)



## 👥 Team Members

- Guru Deepak
- Ramya Chandra
- Rohith Reddy
- Hemanth Reddy



*For issues or contributions, please open a GitHub issue or submit a pull request.*

