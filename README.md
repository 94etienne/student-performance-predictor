# 🎓 Student Performance Predictor

A machine learning web application that predicts student academic performance (Pass/Fail) based on various factors like study hours, attendance, participation, and parental support.

## 🚀 Live Demo

**Streamlit Cloud:** [https://your-username-student-performance-predictor-app-streamlit-app.streamlit.app/](https://your-username-student-performance-predictor-app-streamlit-app.streamlit.app/)

## 📋 Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Real-time Predictions**: Instant Pass/Fail predictions with confidence scores
- **Probability Analysis**: Detailed breakdown of prediction probabilities
- **Smart Recommendations**: Personalized suggestions for improvement
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud

## 📊 Model Features

The model considers the following factors:

- **Study Hours**: Daily study time (0-10 hours)
- **Attendance Rate**: Class attendance percentage (40-100%)
- **Class Participation**: Active vs Passive participation
- **Assignments Submitted**: Number of completed assignments (5-10)
- **Parental Support**: Level of family support (Low/Medium/High)

## 🏃‍♂️ Quick Start

### Option 1: Run Online (Recommended)
Click the live demo link above to use the app immediately without any setup!

### Option 2: Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (first time only)
```bash
python train_model.py
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Open your browser** and go to `http://localhost:8501`

## 📁 Project Structure

```
student-performance-predictor/
│
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── student_model.pkl     # Trained model (generated)
├── columns.csv           # Feature names (generated)
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🎯 How to Use

1. **Access the App**: Open the live demo or run locally
2. **Input Student Data**: Use the sidebar to enter student information:
   - Adjust study hours slider (0-10 hours)
   - Set attendance rate (40-100%)
   - Select participation level (Active/Passive)
   - Choose assignments submitted (5-10)
   - Pick parental support level (Low/Medium/High)
3. **View Prediction**: Get instant Pass/Fail prediction with confidence score
4. **Check Recommendations**: Review personalized improvement suggestions
5. **Analyze Probabilities**: Examine detailed probability breakdown

## 🧠 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~85% on test data
- **Features**: 6 input features with categorical encoding
- **Training Data**: 300 synthetic student records

## 🔧 Model Training Details

The model is trained on synthetic data that simulates realistic student performance patterns:

- **Performance Calculation**: Weighted scoring system based on:
  - Study hours (30% weight)
  - Attendance rate (30% weight) 
  - Class participation (20% weight)
  - Assignment completion (15% weight)
  - Parental support (5% weight)

- **Pass Threshold**: Students scoring ≥60 points are classified as "Pass"

## 📈 Feature Importance

Based on the trained model:
1. **Attendance Rate** - Most important predictor
2. **Study Hours** - Strong correlation with success
3. **Assignment Completion** - Consistent work matters
4. **Class Participation** - Engagement factor
5. **Parental Support** - Environmental influence

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account
2. **Visit [Streamlit Cloud](https://streamlit.io/cloud)**
3. **Connect your GitHub account**
4. **Deploy the app**:
   - Repository: `your-username/student-performance-predictor`
   - Branch: `main`
   - Main file path: `app.py`
5. **Your app will be live** at: `https://your-username-student-performance-predictor-app.streamlit.app/`

### Deploy to Other Platforms

- **Heroku**: Add `Procfile` and deploy
- **Railway**: Connect GitHub repo and deploy
- **Render**: Deploy from GitHub with auto-deploy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **scikit-learn** - For machine learning capabilities
- **Pandas & NumPy** - For data processing

## 📞 Contact

- **GitHub**: [@your-username](https://github.com/your-username)
- **Email**: your.email@example.com

---

**⭐ If you found this project helpful, please give it a star!**