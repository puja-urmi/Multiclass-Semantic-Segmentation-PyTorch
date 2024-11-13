# Multiclass Semantic Segmentation on KiTS19 with PyTorch

This repository provides a deep learning approach for multiclass semantic segmentation on the **Kidney Tumor Segmentation 2019 (KiTS19)** dataset using PyTorch. The project slices 3D CT scans into 2D images, generating 3-channeled masks to classify kidney, tumor, and background regions.

## Project Structure

- **slicer.ipynb**: Jupyter notebook for preprocessing the KiTS19 dataset. This notebook slices the 3D images into 2D slices and generates corresponding 3-channeled 2D masks, with each channel representing:
  - **Background**: No kidney or tumor
  - **Kidney**: Kidney tissue excluding the tumor
  - **Tumor**: Malignant tumor regions within the kidney

- **train.ipynb**: Jupyter notebook for training the segmentation model. Includes:
  - **Data Loading**: Handles data loading for both 2D images and masks.
  - **Training Procedure**: Model training using custom and standard loss functions to optimize for accurate kidney and tumor segmentation.

- **test.ipynb**: Jupyter notebook for evaluating the model's performance. This includes:
  - **Testing**: Loading test data and evaluating the trained model on kidney and tumor segmentation.
  - **Results Visualization**: Visual comparison between ground truth masks and model predictions for qualitative analysis.

- **source/**: Contains all necessary resources, modules, and helper scripts used across training, testing, and data preparation.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies: Listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   gh repo clone puja-urmi/Multiclass-Semantic-Segmentation-PyTorch
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. Download the KiTS19 dataset from the official source, https://github.com/neheller/kits19
2. Place the dataset in the `data/` directory (or adjust paths in `slicer.ipynb` if necessary).
3. Run `slicer.ipynb` to generate 2D images and masks of given 210 cases.
4. Seperate data into train_data, val_data and test_data. 

## Training the Model

Open `train.ipynb` and follow the instructions to train the model on your dataset. Ensure you have sufficient GPU resources for faster training.

## Testing and Evaluation

To evaluate the model's performance, run `test.ipynb`. This notebook loads the trained model and applies it to the test set for kidney and tumor segmentation. Visualizations and performance metrics are included for easy interpretation.

## Results
- **Quantitative Results**:
Kidney - Mean Dice: 91.0318
Kidney - Mean Jaccard: 85.7883
Kidney - Mean Sensitivity: 90.3453
Kidney - Mean Precision: 94.9790
Tumor - Mean Dice: 62.8176
Tumor - Mean Jaccard: 53.1504
Tumor - Mean Sensitivity: 66.7713
Tumor - Mean Precision: 80.7901
Composite Dice (Kidney & Tumor): 76.9247
Composite Jaccard (Kidney & Tumor): 69.4693
Composite Sensitivity (Kidney & Tumor): 78.5583
Composite Precision (Kidney & Tumor): 87.8845
  
- **Qualitative Results**: 

![590](https://github.com/user-attachments/assets/42ffe1ce-04ff-440b-9015-a7645f5b57fc)
![1582](https://github.com/user-attachments/assets/e0f6c149-1ba8-4874-a37d-7faddf91359a)






## Acknowledgments

- KiTS19 Challenge for providing the dataset
- PyTorch community for tools and resources

## License

MIT License
