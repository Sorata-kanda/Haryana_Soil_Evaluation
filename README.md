# 🌱 Soil Health Analysis and Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **Predicting soil health degradation in Haryana using nutrient analysis, environmental factors, and machine learning**

---

## 🔗 Important Links

| Resource                    | Link                                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 📄 **Research Paper**       | [View on Google Docs](https://docs.google.com/document/d/1vkZGs7hvJYitXQK_nCUJPp3_td2ENWFKGWIVuZEUBEU/edit?usp=sharing) |
| 💼 **LinkedIn Profile**     | [Connect with me](https://linkedin.com/in/example)                                                                      |
| 📊 **Project Presentation** | [View Slides](https://docs.google.com/presentation/d/example)                                                           |
| 📁 **Dataset Source**       | [Original Data](https://example.com/dataset)                                                                            |
| 🎥 **Demo Video**           | [Watch on YouTube](https://youtube.com/example)                                                                         |
| 📧 **Contact**              | [jabgraboy.natuer@gmail.com](mailto:jabgraboy.natuer@gmail.com)                                                         |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Workflow](#-project-workflow)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project analyzes soil health data from **Haryana, India** across two time periods (2016-2017 and 2025) to:

- Assess current soil quality using 12 nutrient parameters
- Predict future soil health degradation
- Identify key factors affecting soil quality
- Provide actionable insights for farmers and policymakers

**Key Finding:** 77% of soil samples were classified as "Poor" in 2016-2017, increasing to 82% by 2025, indicating systematic degradation.

---

## ✨ Key Features

### 🔬 **Advanced Data Processing**

- **Semantic parsing** of complex soil data formats (`<280 kg/ha`, `10 to >25 kg/ha`)
- **Weighted scoring system** (1-10 scale) for 12 soil nutrients
- Handles both 2016-2017 and 2025 temporal data

### 📊 **Comprehensive Visualizations**

- 12 interactive visualizations including:
  - Soil health distribution analysis
  - Temporal comparison (2016 vs 2025)
  - Nutrient correlation heatmaps
  - District-level performance rankings
  - Scatter plots showing degradation patterns

### 🤖 **Machine Learning Models**

- **Validation Model**: Confirms scoring system consistency (100% accuracy)
- **Prediction Model**: Predicts future soil health from past data (100% accuracy)
- **Real-time prediction** capability for new soil samples
- Feature importance analysis (Organic Carbon: 71.72% importance)

### 🌍 **Environmental Integration**

- Incorporates **soil moisture** and **temperature** data
- Environmental factors rank #2 and #3 in predictive importance

---

## 🔄 Project Workflow

### **1. Data Collection & Preprocessing**

```
Raw CSV Data (3,850 samples)
    ↓
Semantic Parsing (handles <, >, ranges)
    ↓
Numerical Conversion
    ↓
Quality Validation
```

### **2. Feature Engineering**

```
12 Nutrient Parameters + 2 Environmental Factors
    ↓
Weighted Scoring (1-10 scale)
    ↓
Soil Health Classification (Poor/Moderate/Good)
    ↓
Temporal Analysis (2016-2017 vs 2025)
```

**Scoring Weights:**

- Organic Carbon: **3** (highest priority)
- Nitrogen, Phosphorus, Potassium, pH: **2** each
- Micronutrients, Salinity: **1** each

### **3. Machine Learning Pipeline**

```
Features: 14 (12 nutrients + moisture + temperature)
    ↓
Train/Test Split (80/20, stratified)
    ↓
Random Forest Classifier (class_weight='balanced')
    ↓
Two Models:
  • Validation (2017→2017): Validates scoring
  • Prediction (2017→2025): Predicts future
    ↓
Feature Importance Analysis
    ↓
Real-time Prediction Capability
```

### **4. Visualization & Analysis**

```
Processed Data
    ↓
12 Visualization Types
    ↓
17 High-Resolution PNG Files
    ↓
Insights & Recommendations
```

---

## 📈 Results

### **Soil Health Distribution**

| Category     | 2016-2017     | 2025          | Change  |
| ------------ | ------------- | ------------- | ------- |
| **Poor**     | 77.3% (2,975) | 81.8% (3,150) | +175 ⬆️ |
| **Moderate** | 22.7% (875)   | 18.2% (700)   | -175 ⬇️ |
| **Good**     | 0% (0)        | 0% (0)        | 0       |

### **Key Findings**

1. **Systematic Degradation**: 175 samples degraded from Moderate to Poor, 0 improved
2. **Critical Nutrients Declining**:
   - Manganese: -36.4%
   - Sulphur: -35.3%
   - Organic Carbon: -33.7%
3. **Top Predictive Features**:
   - Organic Carbon: 71.72%
   - Soil Moisture: 5.32%
   - Temperature: 4.84%

### **Model Performance**

| Model          | Purpose                  | Accuracy | F1-Score |
| -------------- | ------------------------ | -------- | -------- |
| **Validation** | Validates scoring system | 100%     | 1.000    |
| **Prediction** | Predicts 2025 from 2017  | 100%     | 1.000    |

---

## 🚀 Installation

### **Prerequisites**

- Python 3.8 or higher
- pip package manager

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/soil-health-analysis.git
cd soil-health-analysis
```

### **Step 2: Install Dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

### **Step 3: Verify Installation**

```bash
python --version  # Should be 3.8+
pip list          # Check installed packages
```

---

## 💻 Usage

### **Quick Start (3 Steps)**

#### **1. Process Raw Data**

```bash
python soil_quality_analysis.py
```

**Output:** `soil_quality_analysis_results.xlsx` (82 columns, 3,850 rows)

#### **2. Generate Visualizations**

```bash
jupyter notebook visualization_analysis.ipynb
```

**Output:** 12 PNG visualization files

#### **3. Train ML Models**

```bash
python Model.py
```

**Output:** 5 PNG model files + real-time prediction capability

---

### **Real-Time Prediction Example**

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load trained model (after running Model.py)
# Create new soil sample
new_sample = pd.DataFrame([{
    'Nitrogen_scale(1-10)': 5.5,
    'Phosphorus_scale(1-10)': 5.5,
    'Potassium_scale(1-10)': 6.5,
    'Organic Carbon_scale(1-10)': 5.5,
    'Soil pH_scale(1-10)': 7.5,
    'Soil Salinity_scale(1-10)': 7.5,
    'Boron_scale(1-10)': 7.5,
    'Iron_scale(1-10)': 7.5,
    'Zinc_scale(1-10)': 7.5,
    'Copper_scale(1-10)': 7.5,
    'Sulphur_scale(1-10)': 7.5,
    'Manganese_scale(1-10)': 7.5,
    'Avg_smlvl_at15cm': 11.5,
    'Temperature_°C': 24.0
}])

# Predict future soil health
prediction = rf_model_pred.predict(new_sample)
print(f"Predicted 2025 Soil Health: {prediction[0]}")
# Output: Poor
```

---

## 📁 Project Structure

```
soil-health-analysis/
│
├── 📄 Dataset for Python - Sheet1.csv          # Raw data (3,850 samples)
├── 📄 soil_quality_analysis.py                 # Data processing script
├── 📄 soil_quality_analysis_results.xlsx       # Processed data (82 columns)
├── 📄 Model.py                                  # ML models + predictions
├── 📓 visualization_analysis.ipynb             # Jupyter notebook (12 viz)
│
├── 📊 Visualizations/ (12 files)
│   ├── 01_soil_health_distribution.png
│   ├── 02_transition_matrix.png
│   ├── 03_nutrient_distributions.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_nutrients_vs_soil_health.png
│   ├── 06_temporal_comparison.png
│   ├── 07_district_analysis.png
│   ├── 08_district_rankings.png
│   ├── 09_nitrogen_vs_organic_carbon.png
│   ├── 10_organic_carbon_vs_soil_health.png
│   ├── 11_temporal_scatter_comparison.png
│   └── 12_pairplot_nutrients.png
│
├── 🤖 ML Models/ (5 files)
│   ├── confusion_matrix_validation.png
│   ├── confusion_matrix_prediction.png
│   ├── feature_importance_validation.png
│   ├── feature_importance_prediction.png
│   └── model_comparison.png
│
├── 📄 README.md                                 # This file
├── 📄 PROJECT_STRUCTURE.md                      # Detailed documentation
├── 📄 QUICK_START.md                            # Quick start guide
└── 📄 .gitignore                                # Git ignore rules
```

---

## 🛠️ Technologies Used

### **Programming & Data Science**

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### **Machine Learning**

- **Scikit-learn** - ML models and evaluation
- **Random Forest Classifier** - Primary ML algorithm

### **Visualization**

- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Jupyter Notebook** - Interactive analysis

### **Data Processing**

- **OpenPyXL** - Excel file handling
- **Regular Expressions** - Semantic parsing

---

## 🎯 Practical Applications

### **1. Early Warning System**

- Test current soil conditions
- Predict future health (5-10 years)
- Intervene before severe degradation

### **2. District-Level Planning**

- Identify at-risk areas
- Allocate resources efficiently
- Monitor intervention effectiveness

### **3. Farmer Advisory**

- Input soil test results
- Get personalized predictions
- Receive nutrient recommendations

### **4. Policy Making**

- Model different scenarios
- Predict long-term outcomes
- Make data-driven decisions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Rahul Jangra**

- LinkedIn: [linkedin.com/in/example](https://linkedin.com/in/example)
- Email: [jabgraboy.natuer@gmail.com](mailto:jabgraboy.natuer@gmail.com)
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **Data Source**: Haryana Soil Health Data (2016-2025)
- **Institution**: [Your University/Institution Name]
- **Supervisor**: [Supervisor Name]
- **Course**: Data Science / Machine Learning Project

---

## 📞 Support

For questions or support:

- 📧 Email: [jabgraboy.natuer@gmail.com](mailto:jabgraboy.natuer@gmail.com)
- 💬 Open an [Issue](https://github.com/yourusername/soil-health-analysis/issues)
- 📖 Check [Documentation](PROJECT_STRUCTURE.md)

---

## 🌟 Star This Repository

If you found this project helpful, please consider giving it a ⭐!

---

<div align="center">

**Made with ❤️ for sustainable agriculture and data science**

[⬆ Back to Top](#-soil-health-analysis-and-prediction-using-machine-learning)

</div>
