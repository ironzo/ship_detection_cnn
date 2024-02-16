# Ship Detection and Analysis Project

## Project Overview
This project focuses on detecting ships in satellite images as part of the [Airbus Ship Detection Challenge on Kaggle](https://www.kaggle.com/c/airbus-ship-detection/overview). The goal is to identify and segment ships in satellite images, which is a crucial task for maritime safety, fishing regulation, and environmental protection. Due to time and resource constraints, the model was trained on 1% of the dataset provided in the challenge.

## Requirements
- pandas
- numpy
- Pillow
- matplotlib
- tensorflow
- scikit-image

To install the required Python modules, run:
```
pip install -r requirements.txt
```

## File Descriptions
- `train.py`: Python script for training the model on ship detection.
- `test.py`: Python script for evaluating the model on a test dataset.
- `eda.ipynb`: Jupyter notebook containing exploratory data analysis (EDA), visualizations, and initial insights into the dataset.

## Setup Instructions
1. Ensure Python 3.6+ is installed on your system.
2. Install the required modules using the `requirements.txt` file.
3. Download and prepare your dataset in the specified directory structure. Due to resource limitations, training was conducted on 1% of the data.

## Usage Guide
- Run `train.py` to start the training process. Customize parameters as needed.
- Use `test.py` to evaluate the trained model's performance on the test dataset.
- Explore `eda.ipynb` for a detailed walkthrough of data preparation, exploration, and initial insights.

## Additional Notes
This project is designed for educational purposes and serves as a template for ship detection tasks. Adjust the model and parameters based on your specific dataset and requirements. The challenge and dataset can be explored further at the [Airbus Ship Detection Challenge on Kaggle](https://www.kaggle.com/c/airbus-ship-detection/overview).