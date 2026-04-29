from flask import Flask, request, jsonify, render_template 
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    encoders = data['encoders']
    train_columns = data['columns']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        # Create DataFrame from input
        input_data = {}
        for col in train_columns:
            val = req_data.get(col, "")
            
            # Type casting based on column
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                try:
                    val = float(val) if val != "" else 0.0
                except ValueError:
                    val = 0.0
                input_data[col] = [val]
            else:
                input_data[col] = [str(val)]
                
        df = pd.DataFrame(input_data)
        
        # Apply encoders
        for col in train_columns:
            if col in encoders:
                le = encoders[col]
                # Map to 'UNSEEN' if value not in classes
                classes = list(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in classes else 'UNSEEN')
                df[col] = le.transform(df[col])
                
        # Predict
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1] # Probability of Churn (Yes=1)
        
        return jsonify({
            'success': True,
            'prediction': 'Yes' if prediction[0] == 1 else 'No',
            'probability': f"{probability * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
