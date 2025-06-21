

# **Multiple Disease Prediction System**

## **Overview**

The **Multiple Disease Prediction System** is a web-based application that predicts the likelihood of **diabetes**, **heart disease**, and **thyroid cancer** using machine learning. Developed with **Python 3.13** and **Streamlit**, the system provides a clean and interactive interface for users to input medical data and receive real-time predictions powered by pre-trained models. The entire environment is managed using the `uv` tool for simplicity and reproducibility.

---

## **Key Features**

* **Diabetes Prediction**
  Predicts diabetes based on features like glucose level, BMI, age, and more.

* **Heart Disease Prediction**
  Uses inputs such as cholesterol, blood pressure, and chest pain type to evaluate risk.

* **Thyroid Cancer Prediction**
  Determines cancer risk using indicators from thyroid hormone function and pathological factors.

* **Streamlit Web Interface**
  Sidebar-based navigation with individual modules for each disease.

* **Pre-Trained ML Models**
  Models are stored as `.sav` files and loaded dynamically for efficient prediction.

---

## **Technologies Used**

| Technology                | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| **Python 3.13**           | Programming language                          |
| **Streamlit**             | Web interface framework                       |
| **Scikit-learn**          | ML model development and inference            |
| **XGBoost**               | Gradient boosting for model accuracy          |
| **Pandas**                | Data handling and manipulation                |
| **NumPy**                 | Numerical operations                          |
| **Matplotlib / Seaborn**  | Data visualization (for model training/EDA)   |
| **Pickle / Pickle5**      | Serialization of trained models               |
| **tqdm**                  | Progress bars for model training or batch ops |
| **uv**                    | Lightweight Python environment management     |
| **streamlit-option-menu** | UI navigation controls for Streamlit apps     |

---

## **Prerequisites**

Before installation, ensure the following tools and libraries are available:

### **Core Requirements**

```bash
Python 3.13
uv
```

### **Install Required Packages**

Use the following to install all project dependencies:

```bash
uv add -r requirements.txt
```

---

## **Installation Guide**

### 1. **Clone the Repository**

```bash
git clone <repository-url>
cd multiple_disease_prediction
```

### 2. **Create and Activate Virtual Environment**

```bash
uv venv
```

* **Windows:**

  ```bash
  .\venv\Scripts\activate
  ```

* **macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

### 3. **Install Dependencies**

Install all packages listed above via `uv add ...` or using a `requirements.txt`.

---

## **Pre-Trained Models & Dataset Files**

Make sure the following models and datasets are available in the correct directories:

### **Model Files**

```
E:\Multiple_Disease_Prediction\saved_models\
├── diabetes_model.sav
├── heart_disease_model.sav
└── thyroid_model.sav
```

### **Datasets Provided**

* `data/diabetes.csv`
* `data/heart_disease_data.csv`
* `data/Thyroid.csv`

You may use these files to retrain the models or understand the data preprocessing pipeline.

---

## **Running the Application**

### **Default Run Command**

```bash
streamlit run src\multiple_disease_pred.py
```

### **Alternatively:**

Run using the command from `streamlit_run_command.txt`:

```bash
streamlit run "E:\Multiple_Disease_Prediction\src\multiple_disease_pred.py"
```

Then open in your browser at:
[http://localhost:8502](http://localhost:8502)

---

## **Project Structure**

```
multiple_disease_prediction/
├── src/
│   └── multiple_disease_pred.py        # Main Streamlit app
├── saved_models/
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── thyroid_model.sav
├── data/
│   ├── diabetes.csv
│   ├── heart_disease_data.csv
│   └── Thyroid.csv
├── streamlit_run_command.txt
└── README.md
```

---

## **Important Notes**

* **Model Paths:**
  Ensure model file paths in `multiple_disease_pred.py` match your folder structure.

* **Input Validation:**
  Users must provide numerical inputs for diabetes and heart disease. Thyroid predictions require specific categorical inputs.

* **Thyroid Encoding:**
  Categorical values must be consistent with the model’s label encoding logic in the preprocessing step.

---

## **Contributing**

We welcome contributions. To contribute:

```bash
# Fork the repository
# Create a new branch
git checkout -b feature-branch

# Make your changes
git commit -m "Add feature"

# Push changes
git push origin feature-branch

# Open a Pull Request
```

---


## **Contact**

📧 **[hamzaimtiaz8668@gmail.com](mailto:your-email@example.com)**
For queries, suggestions, or support, please get in touch.

