# Network Anomaly Detection for DDoS Attacks

## Project Overview
Machine Learning project for **Cisco Bangalore** fresher role, implementing network anomaly detection to identify DDoS attacks using unsupervised learning techniques.

## Results
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Autoencoder (Winner)** | 76.53% | 94.24% | 62.44% | 75.11% | 84.57% |
| Isolation Forest v2 | 70.69% | 82.08% | 61.83% | 70.53% | 75.08% |

## Dataset
- **Source:** CIC-IDS2017 (Friday DDoS dataset)
- **Samples:** 225,745 network flows
- **Features:** 78 numerical features (flow duration, packet stats, flags, etc.)
- **Classes:** BENIGN (43.3%), DDoS (56.7%)

## Technologies Used
- Python 3.12
- TensorFlow 2.20 (GPU-accelerated on RTX 3050 Ti)
- Scikit-learn 1.4.2
- Pandas, NumPy for data preprocessing
- Streamlit for interactive dashboard
- Plotly for visualizations

## Models Implemented

### 1. Autoencoder (Deep Learning)
- Architecture: Encoder (78->64->32->16->8) + Decoder (8->16->32->64->78)
- Training: Only on BENIGN traffic (unsupervised)
- Detection: High reconstruction error = anomaly
- Performance: 75.11% F1-score, 84.57% ROC-AUC

### 2. Isolation Forest (Tree-based)
- Algorithm: Random forest with 200 trees
- Training: Only on BENIGN traffic
- Detection: Short isolation path = anomaly
- Performance: 70.53% F1-score, 75.08% ROC-AUC

## How to Run

### Prerequisites
pip install pandas numpy scikit-learn tensorflow streamlit plotly

### Train Models
jupyter notebook anomaly_detection_model.ipynb

### Launch Dashboard
streamlit run streamlit_app.py

### Technical Highlights
- Feature Engineering: 78 network flow features (SYN/ACK flags, byte rates)
- Class Imbalance Handling: Trained on normal traffic only
- Threshold Optimization: ROC curve analysis
- GPU Acceleration: TensorFlow GPU support
- Explainability: Reconstruction error analysis

## Future Enhancements
- Real-time PCAP file upload and analysis
- SHAP explainability for feature importance
- Multi-class detection (Infiltration, Port Scan, Web Attacks)
- Federated learning for distributed threat intelligence
- Docker containerization for cloud deployment

## Author
**Prajwal S Tirthahalli**
Computer Science & Design Student
ML/Data Science Enthusiast


## Acknowledgments
- Dataset: Canadian Institute for Cybersecurity (CIC-IDS2017)
- Inspiration: Cisco Talos threat intelligence and Firepower NGFW
- Tools: TensorFlow, Scikit-learn, Streamlit communities

---
