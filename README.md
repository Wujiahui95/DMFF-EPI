# DMFF-EPI: Dual-Modality Feature Fusion Network with Contrastive Learning for Enhancer-Promoter Interaction Prediction

## Overview
DMFF-EPI is a deep learning framework designed to predict enhancer-promoter interactions (EPIs) by integrating **dynamic sequence features** (captured via CNN-BiGRU) and **static statistical features** (processed via k-mer frequency MLP). The core innovation is a **contrastive learning regularizer** that aligns complementary biological representations, significantly enhancing prediction accuracy and cross-cell-line generalization.

## Key Features
- **Dual-Modality Fusion**:
  - **Dynamic Path**: CNN-BiGRU extracts local motifs and long-range dependencies from sequences.
  - **Static Path**: MLP encodes global k-mer compositional biases.
- **Contrastive Alignment**:
  - Minimizes distance between dynamic and static features of the same sample.
  - Maximizes distance for different samples.
- **Joint Optimization**:
  - Combines binary cross-entropy loss for classification with contrastive loss for representation alignment.

## Dataset
- **Source**: TargetFinder benchmark.
- **Cell Lines**: GM12878, HUVEC, HeLa, IMR90, K562, NHEK.
- **Preprocessing**:
  - Convert sequences to 6-mer tokens.
  - Generate DNA embeddings using DNA2Vec.
  - Compute k-mer frequency vectors.
- **Data Structure**:
   -  Data/{cell_line}/
    - ├── {cell_line}_train.npz # X_en_tra, X_pr_tra, y_tra
    - └── {cell_line}_test.npz # X_en_tes, X_pr_tes, y_tes
 

## Dependencies
- Python 3.8+
- PyTorch 2.0.0
- NumPy
- Pandas
- scikit-learn

## Usage

### 1. Data Preparation
Generate frequency vectors using the provided script:

```python
from train_torch import compute_avg_row_frequencies

# Example for GM12878 training data
compute_avg_row_frequencies(X_en_tra, save_path='X_en_tra_GM12878.npz')
```
You can fetch the file {Data} to get the datasets to train directly.

### 2. Training
Run the training script with customizable arguments:



  - python train_torch.py --cell_line K562 --epochs 30 --batch_size 16

- Arguments:
  - cell_line: Target cell line (default: K562).

  - temperature: Contrastive loss temperature (default: 0.5).

  - embedding_path: Path to DNA2Vec embedding matrix (default: embedding_matrix.npy).

- Outputs:

  - Trained models saved to: ./model/specificModel/{cell_line}Model{epoch}.pth.

  - Validation metrics logged per epoch.

### 3. Evaluation
Evaluate the model on test data:
```python
python test_torch.py --model_path ./model/specificModel/K562Model.pth
```
  - Outputs:

    - AUC and AUPR metrics for all cell lines.

    - Prediction results saved to: test_sample_{cell_line}_results.csv.

  - Results
    
      | Cell Line | AUROC   | AUPR    |
      |-----------|---------|---------|
      | GM12878   | 0.9569  | 0.8598  |
      | HUVEC     | 0.9632  | 0.7693  |
      | NHEK      | 0.9929  | 0.9072  |
      | HeLa      | 0.9582  | 0.8649  |





  - Key Findings
    - Outperformed baselines (SPEID, EPIANN, SIMCNN) in 5/6 cell lines.

    - Average AUROC improvement of 2.04% over the best baseline.

    - Contrastive learning contributes +2.36% AUROC (based on ablation studies).
