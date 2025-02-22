from Bio import SeqIO
import numpy as np
import pandas as pd
from collections import Counter
import streamlit as st
from io import StringIO
import json

def load_genomic_data(file):
    """Load genomic data from uploaded file or use sample data"""
    sequences = []
    names = []
    try:
        # Create temporary file to handle Streamlit's UploadedFile
        content = file.getvalue().decode('utf-8')
        for record in SeqIO.parse(StringIO(content), "fasta"):
            sequences.append(str(record.seq))
            names.append(record.id)
        return pd.DataFrame({'id': names, 'sequence': sequences})
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_sample_data():
    """Load sample genomic data for demonstration"""
    sequences = [
        "ATGCTAGCTAGCTAG",
        "GCTAGCTAGCTAGCT",
        "TAGCTAGCTAGCTAG",
        "CTAGCTAGCTAGCTA"
    ]
    names = [f"Sample_Sequence_{i+1}" for i in range(len(sequences))]
    return pd.DataFrame({'id': names, 'sequence': sequences})

def extract_kmers(sequence, k=6):
    """Extract k-mers from sequence"""
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return kmers

def get_sequence_stats(sequences_df):
    """Calculate basic sequence statistics"""
    stats = {
        'Total Sequences': len(sequences_df),
        'Average Length': int(sequences_df['sequence'].str.len().mean()),
        'Min Length': sequences_df['sequence'].str.len().min(),
        'Max Length': sequences_df['sequence'].str.len().max(),
        'GC Content (%)': calculate_gc_content(sequences_df['sequence'])
    }
    return stats

def calculate_gc_content(sequences):
    """Calculate average GC content of sequences"""
    gc_content = []
    for seq in sequences:
        gc = sum(seq.count(x) for x in ['G', 'C'])
        total = len(seq)
        gc_content.append((gc/total) * 100)
    return round(np.mean(gc_content), 2)

def process_sequences(sequences_df, k=6):
    """Process sequences and extract features"""
    all_kmers = []
    for seq in sequences_df['sequence']:
        kmers = extract_kmers(seq, k)
        all_kmers.extend(kmers)

    # Get kmer frequencies
    kmer_freq = Counter(all_kmers)
    return kmer_freq

def export_results(sequences_df, model_metrics, kmer_frequencies):
    """Export analysis results to JSON"""
    results = {
        'sequences': sequences_df.to_dict(orient='records'),
        'model_metrics': model_metrics,
        'kmer_frequencies': dict(kmer_frequencies)
    }
    return json.dumps(results, indent=2)