# 🎓 Student Marks Score Prediction System

A Machine Learning project developed using **Python** to predict a student's **exam score (0–100)** based on various academic and personal factors. The project follows the complete Machine Learning pipeline—from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, and deployment using **Streamlit**.

---

## 📌 Project Overview

The **Student Marks Score Prediction System** is a regression-based Machine Learning application that predicts a student's expected exam score using historical student data. The model learns relationships between students' study habits and their exam performance, then estimates the exam score for new students.

The project demonstrates the complete Machine Learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, model evaluation, and deployment through an interactive Streamlit web application.

---

## 🎯 Objectives

- Predict students' exam scores on a scale of **0 to 100**.
- Analyze how different lifestyle and academic factors affect student performance.
- Compare multiple regression algorithms to identify the best-performing model.
- Build an easy-to-use web application for real-time score prediction.
- Demonstrate an end-to-end Machine Learning workflow.

---

## ❗ Problem Statement

Educational institutions often struggle to identify students who may perform poorly before examinations. Since academic performance is influenced by multiple factors such as study habits, attendance, sleep, and mental health, predicting student performance manually is difficult.

This project addresses the problem by using Machine Learning to analyze these factors and predict a student's expected exam score, enabling timely academic support and better learning outcomes.

---

## ✨ Features

- 📊 Predicts student exam score (0–100)
- 📈 Exploratory Data Analysis (EDA)
- 🧹 Data preprocessing and cleaning
- 🤖 Comparison of multiple regression models
- 📉 Model evaluation using RMSE and R² Score
- 🌐 Interactive Streamlit web application
- 💡 Instant score prediction based on user input

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning Libraries
- Scikit-learn
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Model Storage
- Joblib

### Web Framework
- Streamlit

### Development Tools
- Visual Studio Code
- Jupyter Notebook
- Git & GitHub

---

## 📁 Project Structure

```text
STUDENT-MARKS-SCORE-PREDICTION/
│
├── app/
│   └── app.py
│
├── data/
│   └── student_habits_performance.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── notebook.ipynb
│
├── screenshots/
│   └── output.png
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 📥 Input Features

- Study Hours Per Day
- Attendance (%)
- Mental Health Rating
- Sleep Hours
- Part-Time Job (Yes/No)

---

## 📤 Output

- **Predicted Exam Score (0–100)**

---

## 🔄 Project Workflow

1. Collect Student Dataset
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Train-Test Split
6. Train Multiple Regression Models
7. Evaluate Model Performance
8. Select the Best Model
9. Save the Trained Model
10. Deploy using Streamlit

---

## 🤖 Machine Learning Algorithms

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best-performing model is saved as:

```text
best_model.pkl
```

---

## 📊 Evaluation Metrics

- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

The model with the lowest RMSE and highest R² Score is selected for deployment.

---

## 🖥️ Application Workflow

1. User enters:
   - Study Hours Per Day
   - Attendance
   - Mental Health Rating
   - Sleep Hours
   - Part-Time Job
2. The application preprocesses the input.
3. The trained Machine Learning model predicts the student's exam score.
4. The predicted score (0–100) is displayed.

---

## 📊 Sample Input

| Feature | Value |
|---------|------:|
| Study Hours Per Day | 6 |
| Attendance | 92% |
| Mental Health Rating | 8 |
| Sleep Hours | 7 |
| Part-Time Job | No |

---

## 📈 Sample Output

```text
Predicted Exam Score : 88.4
```

---

## 🚀 Future Enhancements

- Student Login System
- Teacher Dashboard
- Performance Analytics
- PDF Report Generation
- Email Notifications
- Cloud Deployment
- College ERP Integration

---

## 📚 Learning Outcomes

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Training
- Model Comparison
- Model Evaluation
- Model Deployment
- Streamlit Web Application Development

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 📸 Project Screenshot

```md
![Application Screenshot](screenshots/output.png)
```

---

## 👨‍💻 Author

**Name:** Anuj

**Course:** Artificial Intelligence & Machine Learning (Python)

**Institution:** Your College Name

---

## 📄 License

This project is developed for educational and learning purposes. It may be modified and extended for academic, research, and non-commercial use.
