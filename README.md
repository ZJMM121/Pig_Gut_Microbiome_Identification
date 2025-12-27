# Pig Gut Microbiome Knowledge Graph Analysis Pipeline

A comprehensive pipeline for constructing and analyzing a pig gut microbiome knowledge graph, including entity standardization, feature vectorization, model training, and prediction validation.

## Pipeline Overview

The pipeline consists of four main stages: **Graph Standardization**, **Feature Vectorization**, **HGNNs Training & Prediction**, and **Prediction Result Validation**.

---

## 1. Graph Standardization

The first step involves standardizing the constructed knowledge graph to ensure consistency and quality of entities and relationships. This process includes:
- Unifying entity identifiers across different data sources
- Normalizing relationship types
- Removing duplicate or erroneous entries
- Establishing consistent data formats for subsequent processing

---

## 2. Feature Vectorization

Entity-specific feature vectors are constructed based on their biological properties:

- **Bacteria**  
  Based on the NCBI Taxonomy system, extract the lineage path of each bacterium. Construct lineage feature vectors (dimension: 1164) using the semantic similarity method proposed by Wang et al. This method effectively reflects phylogenetic similarities between different bacterial species and helps model bacteria that are structurally adjacent but not explicitly mentioned in text.

- **Gene**  
  Construct a semantic network using functional annotations from MeSH and GO data. Calculate the Wang semantic similarity matrix and extract similarity features for each gene relative to others. The resulting vector (dimension: 10474) represents its position in the biological function network.

- **Metabolite**  
  Based on SMILES expressions from PubChem and ChEBI, compute Morgan (ECFP) fingerprints using RDKit tools as molecular structure feature vectors. These vectors reflect potential chemical activities in metabolic pathways.

- **Pathway / Trait / Taxonomy / Segment / PMID**  
  For these less quantifiable entities, one-hot encoding is used to represent their category information. All one-hot feature files are stored in .npz format and undergo unified standardization.

**Standardization**: All entity feature vectors are standardized using the Z-score method. Initial input dictionaries (x_dict) are constructed separately according to entity types for reading by the heterogeneous graph neural network module.

---

## 3. HGNNs Training & Prediction

In the `HGNNs_train_predict` directory, we train and evaluate several classic heterogeneous graph neural network models for link prediction tasks, including:
- CompGCN
- Simple-HGN
- pyHGT
- Other relation prediction models

The training process involves:
- Loading the standardized graph and feature dictionaries
- Configuring model hyperparameters
- Training models with early stopping based on validation metrics
- Saving the best-performing model checkpoint

---

## 4. Prediction Result Validation

The `predict_result_valid` stage involves:
- Loading the best-trained model
- Generating link predictions on the test set
- Evaluating performance using metrics like MRR (Mean Reciprocal Rank) and AUC (Area Under Curve)
- Analyzing and visualizing prediction results to identify biologically meaningful relationships

---

## Requirements

- Python 3.8+
- RDKit (for metabolite fingerprinting)
- PyTorch & PyTorch Geometric (for GNN implementations)
- DGL (Deep Graph Library) for some heterogeneous graph operations
- NumPy, Pandas for data processing
- Scikit-learn for evaluation metrics

## Usage

1. Prepare and standardize your knowledge graph data
2. Run feature vectorization scripts for each entity type
3. Train models using the training scripts in `HGNNs_train_predict`
4. Evaluate predictions using the validation scripts in `predict_result_valid`

For detailed usage instructions of each module, please refer to their respective documentation.
