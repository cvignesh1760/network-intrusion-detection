# Competition Dataset

This directory contains the dataset files for the Network Intrusion Detection Challenge:

- `train.csv`: Labeled training data (84,713 flow records, 84 columns including `id` and `Target`).
- `test.csv`: Unlabeled test data (33,161 flow records, 83 columns including `id`).
- `sample_submission.csv`: Required prediction format (33,161 rows, 2 columns: `id` and `Target`).

### Target Classes
1. **Benign**: Legitimate, normal network traffic sessions.
2. **Availability**: Denial of Service (DoS/DDoS) and flood traffic intended to exhaust server resources.
3. **Reconnaissance**: Probing and port-scanning traffic intended to discover active services.
4. **Spoofing**: Traffic masquerading with forged headers, sequence numbers, or attributes.

### Evaluation Metric
- **Macro-Averaged F1 Score**:
  $$\text{Macro } F_1 = \frac{1}{4} \sum_{c=1}^{4} F_{1, c}$$
  Each class is evaluated independently and averaged equally regardless of class imbalance.
