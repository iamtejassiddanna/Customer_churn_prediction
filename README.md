# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn, complete with an end-to-end training pipeline and a Flask web application for real-time inference.

---

## Project Structure

```
Customer_churn_prediction/
│
├── Customer_churn_prediction.ipynb   # Exploratory data analysis & model experiments
├── Telco Customer Churn.csv          # Dataset (IBM Telco Customer Churn)
│
├── train_model.py                    # Train and serialize the model
├── train.py                          # Alternate training script
├── notebook_code.py                  # Code extracted from notebook
├── extract.py                        # Data extraction utilities
│
├── app.py                            # Flask REST API for predictions
├── templates/                        # HTML frontend (index.html)
├── static/                           # CSS / JS assets
│
└── .gitignore
```

---

## How It Works

1. **Data** — Uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which contains ~7,000 customer records with features like tenure, contract type, payment method, monthly charges, and more.

2. **Training** — `train_model.py` reads the CSV, applies `LabelEncoder` to all categorical columns (with an `UNSEEN` fallback for unknown values at inference time), and trains a `LogisticRegression` classifier. The trained model, encoders, and column names are saved together in `model.pkl`.

3. **Serving** — `app.py` loads `model.pkl` and exposes two routes:
   - `GET /` — renders the prediction form
   - `POST /predict` — accepts JSON, preprocesses input using the stored encoders, and returns a churn prediction (`Yes`/`No`) along with a probability score.

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/iamtejassiddanna/Customer_churn_prediction.git
cd Customer_churn_prediction
pip install flask pandas scikit-learn numpy
```

### Train the Model

```bash
python train_model.py
# Output: model.pkl saved to the project root
```

### Run the App

```bash
python app.py
# Flask server starts at http://127.0.0.1:5000
```

Open your browser and navigate to `http://127.0.0.1:5000` to use the prediction form.

---

## API Usage

**Endpoint:** `POST /predict`  
**Content-Type:** `application/json`

**Example Request:**
```json
{
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "InternetService": "Fiber optic"
}
```

**Example Response:**
```json
{
  "success": true,
  "prediction": "Yes",
  "probability": "74.32%"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| ML Library | scikit-learn |
| Model | Logistic Regression |
| Web Framework | Flask |
| Data Processing | pandas, NumPy |
| Serialization | pickle |

---

## Dataset

- **Source:** IBM Telco Customer Churn (via Kaggle)
- **Rows:** ~7,043 customers
- **Target:** `Churn` — whether the customer left within the last month
- **Features:** 19 attributes including demographics, account info, and services subscribed

---

## Key Files

| File | Description |
|---|---|
| `train_model.py` | Full training pipeline — reads data, encodes, trains, and saves `model.pkl` |
| `app.py` | Flask app with `/predict` endpoint |
| `Customer_churn_prediction.ipynb` | Notebook with EDA and model exploration |
| `Telco Customer Churn.csv` | Raw dataset |

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
