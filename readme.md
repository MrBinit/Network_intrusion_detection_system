# CIDDS-001 Dataset Preprocessing and GNN Model

This project focuses on processing the CIDDS-001 dataset for use with a heterogeneous Graph Neural Network (GNN) to classify benign and malicious network flows. The dataset predominantly contains benign traffic, with minority classes like brute-force attacks and ping scans.

## Key Features and Processing

### Data Distribution
- Features like duration, number of packets, and bytes show long-tail distributions.  
- A power transform is applied to make these features Gaussian-like to improve model performance.

### Dataset Characteristics
- **Numerical Features:** Duration, number of packets, bytes, etc.
- **Categorical Features:** Protocols (TCP, UDP, ICMP, IGMP), attack types.
- **Other Features:** Timestamps, IP addresses.

### Preprocessing Steps
1. **One-Hot Encoding:** Applied to categorical features such as protocols and TCP flags (e.g., SYN, FIN).
2. **Timestamp Normalization:** Converted to time-of-day and normalized between 0 and 1.
3. **IP Address Encoding:** Only the last 16 bits of IPv4 addresses are retained and binary-encoded.
4. **Bytes Cleanup:** Fixed non-numeric values by converting 'm' (millions) to numerical values.
5. **Scaling:** Applied `PowerTransformer` to duration, number of packets, and number of bytes.

### Data Splitting
- Train/Validation/Test split: 80% / 10% / 10%.

### Graph Construction
- **Nodes:**
  - **Hosts:** Represent computers using IP address features.
  - **Flows:** Represent connections between hosts and include all other features.
- **Edges:**
  - Host-to-Flow (source).
  - Flow-to-Host (destination).
- Subgraphs: Dataset divided into subgraphs (e.g., 1024 nodes each) for memory efficiency.

### Data Loader
- Maps IP addresses to node indices and creates edge indices for subgraphs.

## Graph Neural Network (GNN) Architecture

### Model Design
- **Node Types:** Hosts and flows.
- **Layers:** Three `SAGEConv` layers with LeakyReLU activations.
- **Output:** A 5-dimensional vector corresponding to each class (e.g., benign, ping scan).
- **Loss Function:** Cross-entropy loss.
- **Optimizer:** Adam (learning rate = 0.001).

### Training and Evaluation
- **Training Loop:** Model trained for 101 epochs.
- **Test Metrics:**
  - Macro-averaged F1 score for handling imbalanced classes.
  - Confusion matrix to analyze class-wise performance.

### Observations
- The model performs well on majority classes (e.g., benign traffic).
- Minority classes (e.g., ping and port scans) are harder to classify due to fewer samples.
- Errors between similar classes (e.g., ping vs. port scans) highlight potential areas for feature engineering and oversampling.

## Future Improvements
- Introduce additional host-related features (e.g., logs, CPU utilization).
- Use oversampling or class weights to improve minority class detection.
- Leverage GNN scalability for processing larger datasets with millions of flows.

## Conclusion
The project demonstrates the effectiveness of a heterogeneous GNN for flow classification, leveraging advanced preprocessing and tailored graph construction. While the model excels in classifying majority classes, future work could enhance minority class detection for more balanced performance.
