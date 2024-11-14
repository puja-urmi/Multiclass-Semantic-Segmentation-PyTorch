# Multiclass Semantic Segmentation on KiTS19 with PyTorch

This repository offers a complete framework for multiclass semantic segmentation on the **Kidney Tumor Segmentation 2019 (KiTS19)** dataset using PyTorch. This project processes 3D CT scans into 2D slices, generating 3-channel masks to classify regions as background, kidney, or tumor.

## Project Structure

- **slicer.ipynb**: Preprocessing notebook that:
  - Converts 3D CT scans into 2D slices.
  - Generates 3-channel masks for each 2D slice with channels representing:
    - **Background** (Channel 1): Regions without kidney or tumor.
    - **Kidney** (Channel 2): Kidney tissue excluding any tumors.
    - **Tumor** (Channel 3): Malignant tumor regions within the kidney.

- **train.ipynb**: Model training notebook that includes:
  - **Data Loading**: Loads 2D slices and corresponding masks.
  - **Training Procedure**: Implements custom and standard loss functions to optimize kidney and tumor segmentation accuracy.

- **test.ipynb**: Evaluation notebook for:
  - **Testing**: Runs the trained model on test data for kidney and tumor segmentation.
  - **Results Visualization**: Provides visual comparisons of ground truth masks vs. predictions for qualitative analysis.

- **source/**: Contains all helper modules and scripts for data preparation, training, and testing.

## Requirements
- Python 3.x
- PyTorch
- Additional dependencies (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   gh repo clone puja-urmi/Multiclass-Semantic-Segmentation-PyTorch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. Download the [KiTS19 dataset](https://github.com/neheller/kits19) and place it in the `data/` directory (adjust paths in `slicer.ipynb` as needed).
2. Run `slicer.ipynb` to generate 2D slices and masks for the 210 cases.
3. Split the data as follows:
   - Training data: 160 cases
   - Validation data: 20 cases
   - Test data: 30 cases

## Model Training

Open `train.ipynb` and follow the instructions to train the model. For best results, run this notebook on a system with sufficient GPU resources.

## Testing and Evaluation

Run `test.ipynb` to evaluate the trained model on the test dataset. This notebook includes visualization of predictions alongside ground truth masks and provides performance metrics for quantitative analysis.

## Results

- **Quantitative Results**:
  - Kidney:
    - Dice: 91.03%
    - Jaccard: 85.79%
    - Sensitivity: 90.35%
    - Precision: 94.98%
  - Tumor:
    - Dice: 62.82%
    - Jaccard: 53.15%
    - Sensitivity: 66.77%
    - Precision: 80.79%
  - Combined (Kidney & Tumor):
    - Dice: 76.92%
    - Jaccard: 69.47%
    - Sensitivity: 78.56%
    - Precision: 87.88%

- **Qualitative Results**: Sample predictions vs. ground truth masks.

   ![Sample 1](https://github.com/user-attachments/assets/f05921c7-b79c-4cbc-bf08-bed0b4666dc8)
  
   ![Sample 2](https://github.com/user-attachments/assets/6fd3da74-1a88-49f9-85ca-99605e414aa7)
  
   ![Sample 3](https://github.com/user-attachments/assets/1fef1ddc-056a-4661-b0e8-764b18deb0aa)
  
   ![Sample 4](https://github.com/user-attachments/assets/8525f52f-00de-430a-b6e5-767cc1ffeb98)  

## Acknowledgments

- Thanks to the KiTS19 Challenge for dataset access.
- PyTorch community for resources and support.

## License

This project is licensed under the MIT License.
