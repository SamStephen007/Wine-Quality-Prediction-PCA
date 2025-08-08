from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model, scaler, PCA
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n=== Form Data Received ===")
        print(request.form)

        # Collect features in the same order as training
        features = [
            float(request.form.get('fixed_acidity', 0)),
            float(request.form.get('volatile_acidity', 0)),
            float(request.form.get('citric_acid', 0)),
            float(request.form.get('residual_sugar', 0)),
            float(request.form.get('chlorides', 0)),
            float(request.form.get('free_sulfur_dioxide', 0)),
            float(request.form.get('total_sulfur_dioxide', 0)),
            float(request.form.get('density', 0)),
            float(request.form.get('pH', 0)),
            float(request.form.get('sulphates', 0)),
            float(request.form.get('alcohol', 0))
        ]

        print("Features:", features)

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Apply the same preprocessing: scale → PCA → predict
        scaled_features = scaler.transform(features_array)
        pca_features = pca.transform(scaled_features)
        prediction = model.predict(pca_features)[0]

        # Assign result text and CSS class
        if prediction == 1:
            result = "Excellent Quality Wine 🍷"
            quality_class = "excellent"
        else:
            result = "Poor Quality Wine ❌"
            quality_class = "poor"

        print("Prediction Result:", result)

        return render_template(
            'index.html',
            prediction_text=result,
            quality_class=quality_class
        )

    except Exception as e:
        print("!!! Error:", str(e))
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            quality_class="error"
        )

if __name__ == "__main__":
    app.run(debug=True)
