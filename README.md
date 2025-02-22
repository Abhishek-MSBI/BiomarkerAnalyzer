---
title: Viral Disease Biomarker Discovery Platform
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.42.2"
app_file: app.py
pinned: false
---

# 🧬 Viral Disease Biomarker Discovery Platform

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/srabhishek/BiomarkerAnalyser)

A powerful platform for analyzing viral genomic sequences and identifying potential biomarkers using machine learning and deep learning approaches.

## Features

- 📊 Interactive data analysis and visualization
- 🧬 AI-powered sequence analysis
- 🧪 Deep learning model training
- 📈 Comprehensive model evaluation
- 💾 Results export functionality

## Technologies Used

- Python 3.11
- Streamlit
- TensorFlow
- Scikit-learn
- Biopython
- Plotly
- Pandas
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/viral-biomarker-discovery.git
cd viral-biomarker-discovery
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Use the interface to:
   - Upload FASTA files or use sample data
   - Perform sequence analysis
   - Train and evaluate models
   - Export results

## Project Structure

```
viral-biomarker-discovery/
├── data/                # Folder for raw and processed data
├── models/             # Folder for saving trained models
├── src/               # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualizations.py
│   └── ai_analysis.py
├── app.py             # Main application file
├── requirements.txt   # Dependencies
└── README.md         # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Creator

ABHISHEK S R

## Acknowledgments

- Built with Streamlit
- Uses TensorFlow for deep learning
- Employs Scikit-learn for machine learning
- Utilizes Biopython for sequence analysis
