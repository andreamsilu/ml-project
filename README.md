# Machine Learning Coursework: Handwritten Digit Classification

This project implements a machine learning system from scratch to classify handwritten digits using the UCI Optical Recognition of Handwritten Digits dataset. The system is built in Python with support for training, evaluation, and reporting.

---

## **Table of Contents**
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Folder Structure](#folder-structure)
6. [Implementation Details](#implementation-details)
7. [License](#license)

---

## **Features**
- Custom implementation of machine learning algorithms, including:
  - Nearest Neighbor (NN) as a baseline.
  - Multi-Layer Perceptron (MLP) with feedforward and backpropagation.
- Two-fold cross-validation for testing.
- Confusion matrix generation and accuracy reporting.

---

## **Requirements**
Ensure the following are installed:
- Python 3.8+ (tested)
- pip (Python package manager)

Python dependencies are listed in `requirements.txt`.

---

## **Installation**

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ml-coursework.git
cd ml-coursework


python3 -m venv venv
# Activate the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

 
#install dependencies
pip install -r requirements.txt

#Prepare the Data
create  folder named data and store the data files as named below
 dataSet1.csv
 dataSet2.csv
#Run the project
  python3 main.py

#folder structure 
ml-coursework/
├── data/                 # Folder for datasets
│   ├── dataSet1.csv
│   ├── dataSet2.csv
│
├── src/                  # Source code
│   ├── __init__.py       # Makes src a package
│   ├── main.py           # Entry point for the application
│   ├── preprocessing.py  # Data preprocessing logic
│   ├── models.py         # Contains the NN and MLP models
│   ├── train.py          # Training logic
│   ├── evaluate.py       # Model evaluation and metrics
│
│
├── reports/              # Generated reports
│   ├── results.txt       # Results and accuracy metrics
│   ├── confusion_matrix.png
│
├── venv/                 # Virtual environment directory
│
├── .gitignore            # Ignore unnecessary files like venv, __pycache__, etc.
├── requirements.txt      # List of dependencies
├── README.md             # Project overview and instructions

  
## **Folders and Their Roles**

### 1. `data/`
This folder contains the dataset files used for training and testing the model. Ensure the following files are placed here:
- `dataSet1.csv`: Training dataset.
- `dataSet2.csv`: Testing dataset.

### 2. `src/`
The `src` folder contains the source code files that implement the core functionality of the project.

#### Files inside `src/`:
- **`__init__.py`**: Marks the folder as a Python package, enabling modules within it to be imported.
- **`main.py`**: The entry point of the project. It orchestrates the entire workflow:
  1. Loads datasets.
  2. Initializes the model.
  3. Trains the model.
  4. Evaluates the model and saves the results.
- **`preprocessing.py`**: Contains functions to preprocess the dataset. Tasks include:
  - Loading the CSV files.
  - Normalizing the data for better model performance.
- **`models.py`**: Defines the machine learning models used in the project:
  - Nearest Neighbor (NN).
  - Multi-Layer Perceptron (MLP) with feedforward and backpropagation.
- **`train.py`**: Implements the training logic for the MLP model:
  - Performs feedforward computation.
  - Applies backpropagation to update model weights.
- **`evaluate.py`**: Includes functions for testing the model and evaluating its performance:
  - Generates metrics like accuracy.
  - Creates a confusion matrix.

### 3. `notebooks/`
Optional folder for Jupyter Notebooks used during exploratory data analysis (EDA). Example:
- **`analysis.ipynb`**: Contains EDA and visualizations to understand the dataset.

### 4. `reports/`
This folder stores the generated reports and evaluation results.
- **`results.txt`**: Text file summarizing the model's accuracy and key metrics.
- **`confusion_matrix.png`**: A visualization of the confusion matrix showing model performance across classes.

### 5. `venv/`
The virtual environment folder for the project. It contains isolated Python dependencies specific to this project. This folder is automatically created when setting up the virtual environment.

---

## **Supporting Files**

- **`.gitignore`**:
  Specifies files and folders to be ignored by Git version control, such as:
  - `venv/`
  - `__pycache__/`

- **`requirements.txt`**:
  Lists all Python libraries and their versions required for the project. Use the command `pip install -r requirements.txt` to install them.

- **`README.md`**:
  Provides an overview of the project, including setup instructions, usage, and folder structure.

---
 
# ml-project
