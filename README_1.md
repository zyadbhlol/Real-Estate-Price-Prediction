# Real Estate Predictive Analytics

A machine learning project on **9,022 real estate listings** (prices in EGP) that does two things:

1. **Regression**: predicts a property's **price** (`Price_EGP`) with Linear Regression.
2. **Classification**: predicts whether a property is in **high demand** (`High_Demand`) with Logistic Regression.

## Dataset

File: `real_estate_predictive_analytics1.csv` (9,022 rows, 14 columns)

| Column | Description |
|---|---|
| `Area_m2` | Property area in square meters |
| `Bedrooms`, `Bathrooms` | Number of rooms |
| `Floor` | Floor number |
| `Property_Age` | Age of the property in years |
| `Distance_Center_km` | Distance from the city center |
| `Property_Type` | Type of property (including Duplex and Villa) |
| `Location` | Area type (including Outskirts, Residential, Suburban) |
| `Furnished`, `Parking` | Yes / No |
| `Price_EGP` | Price in Egyptian pounds (regression target) |
| `High_Demand` | Yes / No (classification target) |

## Workflow

1. **Data cleaning**: filled missing values in `Area_m2` and `Property_Age` with the median, and removed duplicates.
2. **Feature engineering**: grouped `Property_Age` into `New` (up to 5 years), `Moderate` (6 to 15), and `Old` (over 15).
3. **Encoding**: mapped Yes/No columns to 1/0, and one-hot encoded property type, location, and age category.
4. **Scaling**: standardized numeric features with `StandardScaler`.
5. **Feature selection**: kept features with an absolute correlation above 0.10 with the price, plus furnished, parking, and the encoded categories.
6. **Train/test split**: 80% training and 20% testing (`random_state=42`), stratified for the classification task.
7. **Modeling and evaluation**: Linear Regression and Logistic Regression.

## Results

### Regression (price prediction)

| Metric | Value |
|---|---|
| R² | 0.711 |
| MAE | 698,157 EGP |
| RMSE | 1,000,238 EGP |

The model explains about **71%** of the variation in property prices.

### Classification (high demand)

| Metric | Value |
|---|---|
| Accuracy | 0.756 |
| Precision | 0.766 |
| Recall | 0.849 |
| F1 Score | 0.806 |

Confusion matrix on the test set (1,805 properties):

|  | Predicted Low | Predicted High |
|---|---|---|
| **Actual Low** | 450 | 279 |
| **Actual High** | 162 | 914 |

The model is better at catching high-demand properties (recall 0.85) than low-demand ones (recall 0.62).

## Visualizations

- Correlation heatmap of all features
- Confusion matrix for the classifier
- Feature importance chart (Linear Regression coefficients)

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## How to Run

1. Clone the repository.
2. Install the requirements:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Put `real_estate_predictive_analytics1.csv` in the same folder as the notebook.
4. Open `predictive_model_final.ipynb` and run all cells.

## Author

**zyadbhlol**: [GitHub](https://github.com/zyadbhlol)
