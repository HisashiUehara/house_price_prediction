# House Price Prediction Model Development Project

## Project Overview

This project develops a machine learning model to predict house prices using data from the Ministry of Land, Infrastructure, Transport and Tourism's Real Estate Transaction Price Information System. We implemented both linear regression and neural network approaches for price prediction.

## Data

- **Data Source**: Ministry of Land, Infrastructure, Transport and Tourism "Real Estate Information Library"
- **Region**: Koishikawa, Bunkyo Ward, Tokyo
- **Period**: Q3-Q4 2024
- **Data Size**: 15 records × 28 columns
- **File**: `data/Tokyo_Bunkyo Ward_Koishikawa_20243_20244.csv`

## Development Environment

- **Python**: 3.11.6
- **Key Libraries**:
  - pandas (2.3.1)
  - scikit-learn (1.7.0)
  - matplotlib (3.10.3)
  - numpy (2.3.1)

## Project Structure

```
house_price_prediction/
├── data/                          # Data files
│   └── Tokyo_Bunkyo Ward_Koishikawa_20243_20244.csv
├── venv/                          # Python virtual environment
├── data_analysis.py               # Data analysis script
├── linear_regression_model.py     # Linear regression model
├── neural_network_model.py        # Neural network model
├── simple_csv_test.py             # CSV reading test
├── simple_visualize.py            # Visualization script
└── README.md                      # This file
```

## Execution Steps

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required libraries
pip install pandas scikit-learn matplotlib numpy
```

### 2. Data Analysis

```bash
python data_analysis.py
```

### 3. Linear Regression Model

```bash
python linear_regression_model.py
```

### 4. Neural Network Model

```bash
python neural_network_model.py
```

## Results

### Data Analysis Results

- **Data Size**: 15 records × 28 columns
- **Correlation Coefficient**: 0.637 (moderate positive correlation)
- **Area Range**: 45㎡ - 250㎡
- **Price Range**: 62M - 330M yen

### Model Performance Comparison

| Model | R² Score | RMSE | Evaluation |
|-------|----------|------|------------|
| Linear Regression | -3.033 | 96,261,766 yen | Low prediction accuracy |
| Neural Network | -4.934 | 116,759,806 yen | Low prediction accuracy |

### Key Findings

1. **Insufficient Data**: 15 records are insufficient for machine learning model training
2. **Single Feature Limitation**: Price prediction is difficult with area alone
3. **Complex Factors**: House prices depend on many factors beyond area

## Lessons Learned

### Technical Learnings

1. **Encoding Issues**: Japanese public data often uses cp932 encoding
2. **Data Volume Importance**: Machine learning requires sufficient data volume
3. **Feature Engineering**: Proper feature selection is crucial
4. **Model Selection**: Choose appropriate algorithms based on data characteristics

### Practical Learnings

1. **Theory vs Practice Gap**: Real projects face various unexpected issues
2. **Iterative Improvement**: Building perfect models in one attempt is difficult
3. **Data Quality**: Data preprocessing and validation are essential

## Future Improvements

1. **Increase Data Volume**: Collect more data
2. **Add Features**: Include building age, station distance, building structure, etc.
3. **Try Different Methods**: Test Random Forest, SVM, etc.
4. **Data Preprocessing**: Implement more detailed preprocessing and normalization

## Reflections

1. **Data Collection**: Should have collected more data initially
2. **Feature Design**: Should have considered multiple features from the start
3. **Model Evaluation**: Should have used more detailed evaluation metrics

## What We Learned

- Experienced the actual flow of machine learning projects
- Understood the complexity of data science
- Recognized the importance of iterative improvement
- Realized the difficulty of building practical models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

[Your Name]
