import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load model and feature names
@st.cache_resource
def load_model_and_features():
    try:
        # Check if files exist
        if not os.path.exists("student_model.pkl"):
            st.error("Model file 'student_model.pkl' not found. Please run the training script first.")
            return None, None
        
        if not os.path.exists("columns.csv"):
            st.error("Feature names file 'columns.csv' not found. Please run the training script first.")
            return None, None
        
        # Load model
        with open("student_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load feature names
        feature_names = pd.read_csv("columns.csv", header=None)[0].tolist()
        
        return model, feature_names
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, feature_names = load_model_and_features()

# Title and description
st.title("🎓 Student Performance Predictor")
st.markdown("""
This app predicts whether a student will **Pass or Fail** based on their academic and behavioral characteristics.
Enter the student details in the sidebar to get a prediction.
""")

if model is None or feature_names is None:
    st.stop()

# Display expected features
with st.expander("ℹ️ Model Information"):
    st.write("**Expected Features:**")
    st.write(feature_names)

# Sidebar for input
st.sidebar.header("📝 Input Student Details")
st.sidebar.markdown("Adjust the sliders and dropdowns to enter student information:")

def get_user_input():
    """Get user input from sidebar"""
    
    # Input fields
    study_hours = st.sidebar.slider(
        "📚 Study Hours Per Day", 
        min_value=0.0, 
        max_value=10.0, 
        value=5.0, 
        step=0.5,
        help="Average hours spent studying per day"
    )
    
    attendance = st.sidebar.slider(
        "📅 Attendance Rate (%)", 
        min_value=40, 
        max_value=100, 
        value=80,
        help="Percentage of classes attended"
    )
    
    participation = st.sidebar.selectbox(
        "🗣️ Class Participation", 
        options=[0, 1],
        format_func=lambda x: "Active" if x == 1 else "Passive",
        help="Level of participation in class activities"
    )
    
    assignments = st.sidebar.slider(
        "📝 Assignments Submitted", 
        min_value=5, 
        max_value=10, 
        value=8,
        help="Number of assignments submitted out of 10"
    )
    
    parental_support = st.sidebar.selectbox(
        "👨‍👩‍👧‍👦 Parental Support Level", 
        options=['low', 'medium', 'high'],
        index=1,  # Default to 'medium'
        help="Level of support received from parents"
    )
    
    # Create feature dictionary
    data = {
        'study_hours': study_hours,
        'attendance_rate': attendance,
        'participation': participation,
        'assignments_submitted': assignments,
        'parental_support_medium': 1 if parental_support == 'medium' else 0,
        'parental_support_high': 1 if parental_support == 'high' else 0
    }
    
    return data

# Get user input
input_data = get_user_input()

# Create DataFrame with correct column order
input_df = pd.DataFrame([input_data])

# Ensure columns match the training data
try:
    # Reorder columns to match training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # Display input summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Input Summary")
        
        # Create a more readable display
        display_data = {
            "Study Hours/Day": f"{input_data['study_hours']:.1f}",
            "Attendance Rate": f"{input_data['attendance_rate']:.0f}%",
            "Class Participation": "Active" if input_data['participation'] == 1 else "Passive",
            "Assignments Submitted": f"{input_data['assignments_submitted']}/10",
            "Parental Support": "High" if input_data['parental_support_high'] == 1 else ("Medium" if input_data['parental_support_medium'] == 1 else "Low")
        }
        
        for key, value in display_data.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("🔧 Raw Features")
        st.dataframe(input_df, use_container_width=True)
    
    # Make prediction
    st.subheader("🔍 Prediction Results")
    
    try:
        # Get prediction and probabilities
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("**Prediction: PASS** ✅")
                st.balloons()
            else:
                st.error("**Prediction: FAIL** ❌")
        
        with col2:
            confidence = max(probabilities) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        # Display probability breakdown
        st.subheader("📈 Probability Breakdown")
        
        prob_df = pd.DataFrame({
            'Outcome': ['Fail', 'Pass'],
            'Probability': probabilities
        })
        
        # Create columns for better visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(prob_df.set_index('Outcome'))
        
        with col2:
            st.write("**Detailed Probabilities:**")
            st.write(f"• Fail: {probabilities[0]:.3f} ({probabilities[0]*100:.1f}%)")
            st.write(f"• Pass: {probabilities[1]:.3f} ({probabilities[1]*100:.1f}%)")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if prediction == 0:  # Fail prediction
            recommendations = []
            if input_data['study_hours'] < 4:
                recommendations.append("📚 Increase daily study hours to at least 4 hours")
            if input_data['attendance_rate'] < 80:
                recommendations.append("📅 Improve attendance rate to above 80%")
            if input_data['participation'] == 0:
                recommendations.append("🗣️ Increase participation in class activities")
            if input_data['assignments_submitted'] < 8:
                recommendations.append("📝 Submit more assignments consistently")
            
            if recommendations:
                st.warning("Areas for improvement:")
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.info("The model predicts failure, but your metrics look good. Consider reviewing study methods and seeking additional support.")
        
        else:  # Pass prediction
            st.success("Great! The model predicts success. Keep up the good work!")
            
            # Highlight strengths
            strengths = []
            if input_data['study_hours'] >= 5:
                strengths.append("📚 Good study habits")
            if input_data['attendance_rate'] >= 85:
                strengths.append("📅 Excellent attendance")
            if input_data['participation'] == 1:
                strengths.append("🗣️ Active class participation")
            if input_data['assignments_submitted'] >= 8:
                strengths.append("📝 Consistent assignment submission")
            
            if strengths:
                st.info("Your strengths:")
                for strength in strengths:
                    st.write(f"• {strength}")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        
        # Debug information
        with st.expander("🐛 Debug Information"):
            st.write("**Model Expected Features:**")
            st.write(feature_names)
            st.write("\n**Input DataFrame Columns:**")
            st.write(list(input_df.columns))
            st.write("\n**Input Data:**")
            st.write(input_df)
            st.write("\n**Error Details:**")
            st.write(str(e))

except Exception as e:
    st.error(f"Error processing input: {str(e)}")
    
    with st.expander("🐛 Debug Information"):
        st.write("**Error Details:**")
        st.write(str(e))
        st.write("\n**Expected Features:**")
        st.write(feature_names)
        st.write("\n**Input Data:**")
        st.write(input_data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | 
    <a href="https://streamlit.io" target="_blank">Learn more about Streamlit</a>
    </p>
</div>
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Student Performance Prediction System - Powered by Etienne NTAMBARA</p>
    <p>For support and inquiries, please contact: +250 783 716 761.</p>
</div>
""", unsafe_allow_html=True)
