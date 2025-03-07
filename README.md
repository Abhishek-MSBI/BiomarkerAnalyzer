# GENTIS: Graph-Enhanced Neural Transformer for Integrative Sequence Analysis

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)

## Overview

GENTIS is a novel computational framework that combines positional sequence analysis, conservation features, and transformer-based machine learning to identify biomarkers across diverse biological datasets. By integrating statistical approaches with deep learning, GENTIS enables more robust biomarker detection in complex genomic data.

The framework incorporates negative binomial generalized linear models with transformer-based feature extraction to enhance sensitivity while maintaining specificity. Our architecture allows for the automated processing of sequence data, feature engineering, model training, and visualization within a unified computational pipeline, significantly reducing the computational burden of iterative biomarker analysis.

## Key Features

### Framework Architecture

- **Sequence preprocessing module** that handles data cleaning, normalization, and quality control
- **Feature engineering pipeline** with both traditional biological feature extraction and deep learning-based approaches
- **Statistical analysis module** incorporating negative binomial generalized linear models for count data
- **Transformer-based sequence analysis** using domain-specific pre-trained models
- **Integrated biomarker discovery pipeline** that combines statistical and AI-based approaches
- **Interactive visualization components** for result interpretation

### Data Processing & Analysis

- Support for DNA/RNA/Protein sequences
- Automatic sequence type detection
- Advanced k-mer analysis
- Population and Individual-level analysis
- Integrated genomics datasets
- Real-time sequence validation

### Machine Learning Capabilities

- Multiple training modes:
  - Basic Training
  - Hyperparameter Tuning
  - Ensemble Learning
- Custom CNN architectures
- Transfer learning with transformers
- Cross-validation support
- Model serialization

### Biomarker Discovery

- Automated biomarker detection
- Feature importance analysis
- Motif enrichment analysis
- Conservation analysis
- Mutation hotspot detection
- Functional element prediction

### Visualizations

- Interactive sequence length distribution
- Nucleotide composition plots
- K-mer frequency analysis
- ROC curves
- Confusion matrices
- Feature importance plots
- Training history visualization
- Biomarker network graphs

### Model Evaluation

- Comprehensive metrics (Accuracy, Precision, Recall, F1)
- ROC curve analysis
- Cross-validation results
- Model calibration
- Prediction distribution analysis
- Error analysis

## Technologies Used

- Python 3.8+
- Streamlit
- TensorFlow
- Scikit-learn
- Biopython
- Plotly
- Pandas
- NumPy
- PyTorch & Transformers (for advanced analysis)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

The easiest way to install all dependencies is to use our automated installer script:

```bash
python install_dependencies.py
```

This script will detect your system configuration and install the appropriate versions of all required packages.

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/GENTIS.git
   cd GENTIS
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting Dependencies

If you encounter warnings about missing libraries when running the application:

```bash
# For TensorFlow
pip install tensorflow==2.14.0

# For PyTorch and Transformers
pip install torch==2.0.1 transformers==4.31.0

# For UMAP (dimensionality reduction)
pip install umap-learn==0.5.3
```

## Running the Application

After installation, launch the interactive interface:

```bash
streamlit run app.py
```

## Example Usage

1. Load sample data or upload your own FASTA files
2. Explore sequence characteristics through visualization tools
3. Configure and train the GENTIS model to identify potential biomarkers
4. Evaluate model performance and analyze discovered biomarkers
5. Export results and visualizations for further analysis or publication

## Project Structure

```
GENTIS/
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

## Framework Details

### Enhanced Sequence Processing

The sequence preprocessing module implements robust handling of various biological sequence formats. The `EnhancedSequenceProcessor` class performs quality filtering, length normalization, and validation to ensure that only high-quality sequences enter the analysis pipeline.

### Feature Engineering

GENTIS implements multi-level feature engineering, extracting both traditional sequence features and advanced pattern-based features:

- **Position-specific features**: Implementation of position weight matrices to capture the importance of specific nucleotides or amino acids at each position in aligned sequences.
- **Conservation features**: Calculation of conservation scores across aligned sequences to identify regions under evolutionary constraint.
- **Positional features**: Extraction of local context around potential biomarker positions using sliding window approaches.

### Statistical Modeling and Analysis

For statistical rigor, GENTIS incorporates a `NegativeBinomialGLM` class that properly models count data with overdispersion. The statistical pipeline includes:

- Multiple testing correction using either Benjamini-Hochberg FDR or Bonferroni methods
- Fold-change thresholding with configurable parameters
- Pathway enrichment analysis to identify functional patterns among discovered biomarkers

### Transformer-Based Sequence Analysis

A key innovation in GENTIS is the integration of transformer models for biological sequence analysis. The `GenomicTransformer` class implements attention-based sequence encoding that captures long-range dependencies within biological sequences.

The transformers can be used in three distinct modes:
- Embedding Mean: Using the average of sequence embeddings
- Attention Patterns: Leveraging transformer attention weights
- Learned Features: Extracting learned feature representations

### Integrated Biomarker Pipeline

The `IntegratedBiomarkerPipeline` class represents the core of GENTIS, unifying statistical and deep learning approaches with configurable weighting. This allows researchers to balance the contributions of traditional statistics and AI-based pattern recognition according to the characteristics of their dataset.

## Limitations and Future Work

While GENTIS provides a powerful framework for biomarker discovery, several limitations remain to be addressed in future work:

- The current implementation focuses primarily on nucleotide and protein sequences, with limited support for other data modalities.
- The transformer models require significant computational resources for training on large datasets.
- Integration with external biological knowledge bases could further enhance the biological relevance of discovered biomarkers.

Future development will focus on expanding the framework to incorporate additional data modalities, implementing more efficient transformer architectures, and enhancing the integration with biological knowledge bases.

## Contributing

We welcome contributions to improve GENTIS:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add some NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use GENTIS in your research, please cite:

```
GENTIS: Graph-Enhanced Neural Transformer for Integrative Sequence Analysis in Biomarker Discovery.
```

## Acknowledgments

- We thank the bioinformatics community for their valuable feedback and contributions to the development of this framework.
- Built with Streamlit for interactive visualization
- Uses TensorFlow and PyTorch for deep learning implementations
- Employs Scikit-learn for machine learning algorithms
- Utilizes Biopython for sequence analysis
