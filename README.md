# 🌿 Iris Species Prediction App

A simple machine learning web app built using **Streamlit** and **Random Forest Classifier** to predict the species of an Iris flower based on user-inputted flower dimensions.

(screenshot.png)

---

## 🚀 Features

- Interactive sliders to input Sepal and Petal dimensions.
- Predicts species of Iris flower using a trained Random Forest model.
- Real-time predictions with an easy-to-use UI powered by Streamlit.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**

---

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/iris-rf-streamlit-app.git
   cd iris-rf-streamlit-app
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**  
   ```bash
   streamlit run ml_streamlit.py
   ```

---

## 📁 Project Structure

```
├── ml_streamlit.py                # Main Streamlit application
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── screenshot.png        # (Optional) App screenshot
```

---

## 📊 Model Details

- **Dataset**: Iris dataset from `sklearn.datasets`
- **Model**: RandomForestClassifier from `sklearn.ensemble`
- **Target**: Classifies flowers into:
  - Setosa
  - Versicolor
  - Virginica

---

## ✨ Example Use

Once the app is running, use the sidebar sliders to set:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The model will output the **predicted species** and show your selected measurements.


---

## 📝 License

This project is licensed under the MIT License.

