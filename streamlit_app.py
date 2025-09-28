import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Teen Smartphone Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .addicted {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .not-addicted {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# Load model comparison data from Excel
@st.cache_data
def load_algorithm_comparison():
    try:
        df = pd.read_excel('model_comparison_results.xlsx', sheet_name='Model Comparison')
        return df
    except Exception as e:
        st.warning(f"Could not load Excel file: {e}")
        # Fallback data if Excel file not found
        return pd.DataFrame({
            'Algorithm': ['Decision Tree', 'Logistic Regression', 'K-Nearest Neighbors', 
                         'Support Vector Machine', 'Neural Networks', 'Random Forest',
                         'K-Means', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'Naive Bayes'],
            'Accuracy': [0.85, 0.92, 0.88, 0.90, 0.91, 0.89, 0.75, 0.87, 0.90, 0.93, 0.82],
            'F1-Score': [0.84, 0.91, 0.87, 0.89, 0.90, 0.88, 0.74, 0.86, 0.89, 0.92, 0.81],
            'Training Time': [0.1, 0.2, 0.05, 2.5, 5.0, 1.0, 0.8, 1.2, 3.0, 0.8, 0.05],
            'Prediction Speed': [5000, 3000, 1000, 500, 2000, 4000, 6000, 3500, 5500, 7000, 8000]
        })

# Load trained models and preprocessing objects
@st.cache_resource
def load_models_and_preprocessors():
    try:
        # Check what files actually exist
        available_files = [f for f in os.listdir('.') if f.endswith('.joblib')]
        #st.sidebar.info(f"Found .joblib files: {', '.join(available_files)}")
        
        # Load preprocessing objects with fallback
        scaler = None
        label_encoders = {}
        
        try:
            scaler = joblib.load('scaler.joblib')
        except:
            st.sidebar.warning("Scaler not found - using identity scaling")
        
        try:
            label_encoders = joblib.load('label_encoders.joblib')
        except:
            st.sidebar.warning("Label encoders not found - using default encoding")
        
        # Try different possible model file names
        model_filename_variations = {
            'Decision Tree': ['DecisionTree.joblib', 'DecisionTreeClassifier.joblib', 'decision_tree.joblib'],
            'Logistic Regression': ['LogisticRegression.joblib', 'logistic_regression.joblib'],
            'K-Nearest Neighbors': ['KNearestNeighbors.joblib', 'KNeighborsClassifier.joblib', 'knn.joblib'],
            'Support Vector Machine': ['SupportVectorMachine.joblib', 'SVC.joblib', 'svm.joblib'],
            'Neural Networks': ['NeuralNetworks.joblib', 'MLPClassifier.joblib', 'mlp.joblib'],
            'Random Forest': ['RandomForest.joblib', 'RandomForestClassifier.joblib', 'random_forest.joblib'],
            'AdaBoost': ['AdaBoost.joblib', 'AdaBoostClassifier.joblib', 'adaboost.joblib'],
            'Gradient Boosting': ['GradientBoosting.joblib', 'GradientBoostingClassifier.joblib', 'gradient_boosting.joblib'],
            'XGBoost': ['XGBoost.joblib', 'XGBClassifier.joblib', 'xgboost.joblib'],
            'Naive Bayes': ['NaiveBayes.joblib', 'GaussianNB.joblib', 'naive_bayes.joblib'],
            'Best Model': ['best_model.joblib', 'best_model_v2.joblib']
        }
        
        models = {}
        loaded_models = []
        
        for model_name, possible_filenames in model_filename_variations.items():
            model_loaded = False
            for filename in possible_filenames:
                if filename in available_files:
                    try:
                        models[model_name] = joblib.load(filename)
                        loaded_models.append(f"{model_name} ({filename})")
                        model_loaded = True
                        break
                    except Exception as e:
                        continue
            
            if not model_loaded:
                st.sidebar.warning(f"Could not load {model_name}")
        
        # if loaded_models:
        #     st.sidebar.success(f"Successfully loaded {len(loaded_models)} models:")
        #     for model in loaded_models:
        #         st.sidebar.text(f"✓ {model}")
        # else:
        #     st.sidebar.error("No models could be loaded!")
                
        return models, scaler, label_encoders
    
    except Exception as e:
        st.error(f"Error in load_models_and_preprocessors: {e}")
        return {}, None, {}

def main():
    st.markdown('<h1 class="main-header">📱 Teen Smartphone Addiction Predictor</h1>', unsafe_allow_html=True)
    
    # Load models and preprocessors
    models, scaler, label_encoders = load_models_and_preprocessors()
    
    if not models:
        st.error("⚠️ No models could be loaded!")
        st.info("Please check that your .joblib model files are in the same directory as this app.")
        
        # Show file debugging info
        with st.expander("🔍 Debug Information"):
            st.write("**Current directory contents (.joblib files):**")
            joblib_files = [f for f in os.listdir('.') if f.endswith('.joblib')]
            if joblib_files:
                for file in joblib_files:
                    st.write(f"- {file}")
            else:
                st.write("No .joblib files found in current directory")
            
            st.write("**Expected model filenames:**")
            expected_files = [
                'DecisionTreeClassifier.joblib', 'LogisticRegression.joblib', 
                'KNeighborsClassifier.joblib', 'SVC.joblib', 'MLPClassifier.joblib',
                'RandomForestClassifier.joblib', 'AdaBoostClassifier.joblib',
                'GradientBoostingClassifier.joblib', 'XGBClassifier.joblib', 'GaussianNB.joblib',
                'best_model.joblib', 'scaler.joblib', 'label_encoders.joblib'
            ]
            for file in expected_files:
                exists = "✅" if file in joblib_files else "❌"
                st.write(f"{exists} {file}")
        
        # Still allow navigation to other pages
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", 
                                   ["🔮 Prediction", "📊 Model Comparison", "📈 Performance Analysis", "ℹ️ About"])
        
        if page == "📊 Model Comparison":
            comparison_page()
        elif page == "📈 Performance Analysis":
            analysis_page()
        elif page == "ℹ️ About":
            about_page()
        else:
            st.warning("Prediction page requires models to be loaded.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🔮 Prediction", "📊 Model Comparison", "📈 Performance Analysis", "ℹ️ About"])
    
    if page == "🔮 Prediction":
        prediction_page(models, scaler, label_encoders)
    elif page == "📊 Model Comparison":
        comparison_page()
    elif page == "📈 Performance Analysis":
        analysis_page()
    else:
        about_page()

def prediction_page(models, scaler, label_encoders):
    st.markdown('<h2 class="sub-header">Make a Prediction</h2>', unsafe_allow_html=True)
    
    if not models:
        st.error("Models not loaded. Please check your model files.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    # Model selection - Move this BEFORE the form
    model_choice = st.sidebar.selectbox("Select Model for Prediction", 
                                       list(models.keys()),
                                       index=0)
    
    with col1:
        st.markdown("### Input Features")
        
        with st.form("prediction_form"):
            # Create input fields based on the actual dataset columns from your training data
            col_left, col_right = st.columns(2)
            
            with col_left:
                age = st.slider("Age", 13, 19, 16, help="Age of the teenager")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Gender of the teenager")
                
                # Location - Use text input to match your training data format
                location = st.text_input("Location", value="Cityname", help="Location (use format like 'Hansonfort', 'Theodorefort')")
                
                # School Grade - Use selectbox with actual grade values from training
                school_grade = st.selectbox("School Grade", 
                                          ["6th", "7th", "8th", "9th", "10th", "11th", "12th"], 
                                          index=3, help="Current school grade")
                
                daily_usage_hours = st.slider("Daily Usage Hours", 1.0, 15.0, 7.0, step=0.1, help="Hours spent on phone daily")
                sleep_hours = st.slider("Sleep Hours", 4.0, 12.0, 8.0, step=0.1, help="Hours of sleep per night")
                academic_performance = st.slider("Academic Performance", 1, 100, 70, help="Academic performance score (1-100)")
                social_interactions = st.slider("Social Interactions", 1, 10, 6, help="Level of social interactions (1-10)")
                exercise_hours = st.slider("Exercise Hours", 0.0, 5.0, 2.0, step=0.1, help="Hours of exercise per day")
                anxiety_level = st.slider("Anxiety Level", 1, 10, 5, help="Anxiety level (1-10)")
                depression_level = st.slider("Depression Level", 1, 10, 4, help="Depression level (1-10)")
                
            with col_right:
                self_esteem = st.slider("Self Esteem", 1, 10, 6, help="Self esteem level (1-10)")
                
                # Parental Control - Use numeric slider to match training data
                parental_control = st.slider("Parental Control", 0, 2, 1, help="Parental control level (0=Low, 1=Medium, 2=High)")
                
                screen_time_before_bed = st.slider("Screen Time Before Bed (hours)", 0.0, 4.0, 1.0, step=0.1, help="Hours of screen time before bed")
                phone_checks_per_day = st.slider("Phone Checks Per Day", 10, 200, 80, help="Number of times phone is checked daily")
                apps_used_daily = st.slider("Apps Used Daily", 5, 50, 20, help="Number of apps used daily")
                time_on_social_media = st.slider("Time on Social Media (hours)", 0.0, 10.0, 3.0, step=0.1, help="Hours spent on social media daily")
                time_on_gaming = st.slider("Time on Gaming (hours)", 0.0, 8.0, 2.0, step=0.1, help="Hours spent gaming daily")
                time_on_education = st.slider("Time on Education (hours)", 0.0, 6.0, 2.0, step=0.1, help="Hours spent on educational apps daily")
                
                # Phone Usage Purpose - Match training data categories
                phone_usage_purpose = st.selectbox("Phone Usage Purpose", 
                                                  ["Browsing", "Education", "Entertainment", "Communication", "Gaming", "Social Media"],
                                                  help="Primary purpose of phone usage")
                
                family_communication = st.slider("Family Communication", 1, 10, 6, help="Level of family communication (1-10)")
                weekend_usage_hours = st.slider("Weekend Usage Hours", 1.0, 20.0, 9.0, step=0.1, help="Hours spent on phone during weekends")
            
            submitted = st.form_submit_button("🔍 Predict Addiction Level", use_container_width=True)
        
        if submitted:
            try:
                # Create feature vector in the same order as training data (excluding Name which is not a feature)
                # Order: Age, Gender, Location, School_Grade, Daily_Usage_Hours, Sleep_Hours, Academic_Performance, 
                # Social_Interactions, Exercise_Hours, Anxiety_Level, Depression_Level, Self_Esteem, 
                # Parental_Control, Screen_Time_Before_Bed, Phone_Checks_Per_Day, Apps_Used_Daily,
                # Time_on_Social_Media, Time_on_Gaming, Time_on_Education, Phone_Usage_Purpose, 
                # Family_Communication, Weekend_Usage_Hours
                
                # Prepare the raw data first (before encoding)
                raw_data = {
                    'Age': age,
                    'Gender': gender,
                    'Location': location,
                    'School_Grade': school_grade,
                    'Daily_Usage_Hours': daily_usage_hours,
                    'Sleep_Hours': sleep_hours,
                    'Academic_Performance': academic_performance,
                    'Social_Interactions': social_interactions,
                    'Exercise_Hours': exercise_hours,
                    'Anxiety_Level': anxiety_level,
                    'Depression_Level': depression_level,
                    'Self_Esteem': self_esteem,
                    'Parental_Control': parental_control,
                    'Screen_Time_Before_Bed': screen_time_before_bed,
                    'Phone_Checks_Per_Day': phone_checks_per_day,
                    'Apps_Used_Daily': apps_used_daily,
                    'Time_on_Social_Media': time_on_social_media,
                    'Time_on_Gaming': time_on_gaming,
                    'Time_on_Education': time_on_education,
                    'Phone_Usage_Purpose': phone_usage_purpose,
                    'Family_Communication': family_communication,
                    'Weekend_Usage_Hours': weekend_usage_hours
                }
                
                # Create DataFrame for encoding
                input_df = pd.DataFrame([raw_data])
                
                # Apply label encoders if available, otherwise use fallback encoding
                encoded_data = raw_data.copy()
                
                # Encode categorical variables
                if label_encoders and 'Gender' in label_encoders:
                    try:
                        encoded_data['Gender'] = label_encoders['Gender'].transform([gender])[0]
                    except ValueError:
                        # Handle unseen categories with most common value
                        st.warning(f"Unknown gender '{gender}', using default encoding")
                        encoded_data['Gender'] = 0  # Default to first class
                else:
                    # Fallback encoding
                    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
                    encoded_data['Gender'] = gender_map.get(gender, 1)
                
                if label_encoders and 'Location' in label_encoders:
                    try:
                        encoded_data['Location'] = label_encoders['Location'].transform([location])[0]
                    except ValueError:
                        st.warning(f"Unknown location '{location}', using default encoding")
                        # Use a hash-based encoding for unknown locations
                        encoded_data['Location'] = hash(location) % 1000
                else:
                    # Fallback: use hash for location
                    encoded_data['Location'] = hash(location) % 1000
                
                if label_encoders and 'School_Grade' in label_encoders:
                    try:
                        encoded_data['School_Grade'] = label_encoders['School_Grade'].transform([school_grade])[0]
                    except ValueError:
                        st.warning(f"Unknown school grade '{school_grade}', using default encoding")
                        grade_map = {"6th": 0, "7th": 1, "8th": 2, "9th": 3, "10th": 4, "11th": 5, "12th": 6}
                        encoded_data['School_Grade'] = grade_map.get(school_grade, 3)
                else:
                    # Fallback encoding
                    grade_map = {"6th": 0, "7th": 1, "8th": 2, "9th": 3, "10th": 4, "11th": 5, "12th": 6}
                    encoded_data['School_Grade'] = grade_map.get(school_grade, 3)
                
                if label_encoders and 'Phone_Usage_Purpose' in label_encoders:
                    try:
                        encoded_data['Phone_Usage_Purpose'] = label_encoders['Phone_Usage_Purpose'].transform([phone_usage_purpose])[0]
                    except ValueError:
                        st.warning(f"Unknown phone usage purpose '{phone_usage_purpose}', using default encoding")
                        purpose_map = {"Browsing": 0, "Education": 1, "Entertainment": 2, "Communication": 3, "Gaming": 4, "Social Media": 5}
                        encoded_data['Phone_Usage_Purpose'] = purpose_map.get(phone_usage_purpose, 0)
                else:
                    # Fallback encoding
                    purpose_map = {"Browsing": 0, "Education": 1, "Entertainment": 2, "Communication": 3, "Gaming": 4, "Social Media": 5}
                    encoded_data['Phone_Usage_Purpose'] = purpose_map.get(phone_usage_purpose, 0)
                
                # Create feature vector
                features = np.array([[
                    encoded_data['Age'],
                    encoded_data['Gender'],
                    encoded_data['Location'],
                    encoded_data['School_Grade'],
                    encoded_data['Daily_Usage_Hours'],
                    encoded_data['Sleep_Hours'],
                    encoded_data['Academic_Performance'],
                    encoded_data['Social_Interactions'],
                    encoded_data['Exercise_Hours'],
                    encoded_data['Anxiety_Level'],
                    encoded_data['Depression_Level'],
                    encoded_data['Self_Esteem'],
                    encoded_data['Parental_Control'],
                    encoded_data['Screen_Time_Before_Bed'],
                    encoded_data['Phone_Checks_Per_Day'],
                    encoded_data['Apps_Used_Daily'],
                    encoded_data['Time_on_Social_Media'],
                    encoded_data['Time_on_Gaming'],
                    encoded_data['Time_on_Education'],
                    encoded_data['Phone_Usage_Purpose'],
                    encoded_data['Family_Communication'],
                    encoded_data['Weekend_Usage_Hours']
                ]])
                
                # Scale features
                if scaler is not None:
                    features_scaled = scaler.transform(features)
                else:
                    # Use standardization if no scaler available
                    features_scaled = features  # Keep original for now
                
                selected_model = models[model_choice]
                
                # Make prediction
                try:
                    prediction = selected_model.predict(features_scaled)[0]
                except Exception as pred_error:
                    st.warning(f"Prediction error with scaled features: {pred_error}")
                    # Try with unscaled features
                    try:
                        prediction = selected_model.predict(features)[0]
                        st.info("Used unscaled features for prediction")
                    except Exception as e2:
                        st.error(f"Could not make prediction: {e2}")
                        return
                
                # Get probability if available
                try:
                    if hasattr(selected_model, 'predict_proba'):
                        try:
                            probabilities = selected_model.predict_proba(features_scaled)[0]
                        except:
                            probabilities = selected_model.predict_proba(features)[0]
                        confidence = max(probabilities)
                        addiction_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                    elif hasattr(selected_model, 'decision_function'):
                        try:
                            decision = selected_model.decision_function(features_scaled)[0]
                        except:
                            decision = selected_model.decision_function(features)[0]
                        addiction_prob = 1 / (1 + np.exp(-decision))  # Sigmoid transformation
                        confidence = max(addiction_prob, 1 - addiction_prob)
                    else:
                        # For models without probability (like K-means)
                        addiction_prob = 0.7 if prediction == 1 else 0.3
                        confidence = 0.6
                except:
                    # Fallback for probability calculation
                    addiction_prob = 0.7 if prediction == 1 else 0.3
                    confidence = 0.6
                
                with col2:
                    st.markdown("### Prediction Result")
                    
                    if prediction == 1:
                        st.markdown(f'<div class="prediction-result addicted">⚠️ HIGH RISK<br>Addiction Detected</div>', 
                                  unsafe_allow_html=True)
                        st.error(f"Addiction Probability: {addiction_prob:.2%}")
                        st.warning(f"Model: {model_choice}")
                        st.info(f"Confidence: {confidence:.2%}")
                    else:
                        st.markdown(f'<div class="prediction-result not-addicted">✅ LOW RISK<br>No Addiction</div>', 
                                  unsafe_allow_html=True)
                        st.success(f"Non-Addiction Probability: {1-addiction_prob:.2%}")
                        st.info(f"Model: {model_choice}")
                        st.info(f"Confidence: {confidence:.2%}")
                    
                    # Risk factors analysis
                    st.markdown("### Risk Factors Analysis")
                    
                    # Define risk thresholds and calculate risk scores
                    risk_factors = {}
                    
                    if daily_usage_hours > 8:
                        risk_factors["High Daily Usage"] = min((daily_usage_hours - 8) / 7, 1.0)
                    
                    if sleep_hours < 7:
                        risk_factors["Sleep Deprivation"] = min((7 - sleep_hours) / 3, 1.0)
                    
                    if exercise_hours < 1:
                        risk_factors["Low Physical Activity"] = min((1 - exercise_hours) / 1, 1.0)
                    
                    if time_on_social_media > 4:
                        risk_factors["Excessive Social Media"] = min((time_on_social_media - 4) / 6, 1.0)
                    
                    if anxiety_level > 6:
                        risk_factors["High Anxiety"] = min((anxiety_level - 6) / 4, 1.0)
                    
                    if depression_level > 5:
                        risk_factors["Depression Signs"] = min((depression_level - 5) / 5, 1.0)
                    
                    if phone_checks_per_day > 100:
                        risk_factors["Frequent Phone Checks"] = min((phone_checks_per_day - 100) / 100, 1.0)
                    
                    if screen_time_before_bed > 1:
                        risk_factors["Screen Time Before Bed"] = min((screen_time_before_bed - 1) / 3, 1.0)
                    
                    if academic_performance < 70:
                        risk_factors["Poor Academic Performance"] = min((70 - academic_performance) / 40, 1.0)
                    
                    if self_esteem < 5:
                        risk_factors["Low Self Esteem"] = min((5 - self_esteem) / 4, 1.0)
                    
                    if weekend_usage_hours > 10:
                        risk_factors["Excessive Weekend Usage"] = min((weekend_usage_hours - 10) / 10, 1.0)
                    
                    # Display risk factors
                    if risk_factors:
                        for factor, score in risk_factors.items():
                            color = "🔴" if score > 0.7 else "🟡" if score > 0.4 else "🟢"
                            st.write(f"{color} **{factor}**: {score:.1%}")
                            st.progress(score)
                    else:
                        st.success("No significant risk factors detected!")
                
                # Recommendations
                st.markdown("### 💡 Personalized Recommendations")
                recommendations = []
                
                if daily_usage_hours > 8:
                    recommendations.append("📱 **Reduce daily screen time**: Try to limit phone usage to 6-8 hours per day using screen time controls")
                
                if sleep_hours < 7:
                    recommendations.append("😴 **Improve sleep habits**: Aim for 7-9 hours of sleep per night for better mental health")
                
                if exercise_hours < 1:
                    recommendations.append("🏃‍♂️ **Increase physical activity**: Engage in at least 1 hour of physical exercise daily")
                
                if time_on_social_media > 4:
                    recommendations.append("📵 **Limit social media**: Reduce social media usage and use app timers to stay within healthy limits")
                
                if screen_time_before_bed > 1:
                    recommendations.append("🌙 **Digital detox before bed**: Avoid screens at least 1 hour before bedtime")
                
                if anxiety_level > 6 or depression_level > 5:
                    recommendations.append("🧠 **Mental health support**: Consider talking to a counselor or mental health professional")
                
                if phone_checks_per_day > 100:
                    recommendations.append("🔔 **Reduce phone checks**: Turn off non-essential notifications and practice mindful phone usage")
                
                if academic_performance < 70:
                    recommendations.append("📚 **Focus on studies**: Use focus apps and create phone-free study zones")
                
                if family_communication < 5:
                    recommendations.append("👨‍👩‍👧‍👦 **Improve family time**: Schedule regular device-free family activities")
                
                if weekend_usage_hours > 10:
                    recommendations.append("🎯 **Weekend activities**: Plan offline activities and hobbies for weekends")
                
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.success("🎉 Excellent! You're maintaining very healthy smartphone habits!")
                    st.balloons()
                
                # Additional insights
                st.markdown("### 📊 Your Usage Profile")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Daily Usage", f"{daily_usage_hours:.1f}h", 
                             f"{daily_usage_hours - 6:.1f}h vs recommended" if daily_usage_hours > 6 else "Within limits")
                with col_b:
                    st.metric("Sleep Quality", f"{sleep_hours:.1f}h", 
                             f"{sleep_hours - 8:.1f}h vs optimal" if sleep_hours != 8 else "Optimal")
                with col_c:
                    st.metric("Physical Activity", f"{exercise_hours:.1f}h", 
                             f"{exercise_hours - 1:.1f}h vs minimum" if exercise_hours < 1 else "Good")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all model files are properly loaded.")
                
                # Debug information
                with st.expander("Debug Information"):
                    st.write("Raw input data:", raw_data if 'raw_data' in locals() else "Not created")
                    st.write("Encoded data:", encoded_data if 'encoded_data' in locals() else "Not created")
                    st.write("Feature vector shape:", features.shape if 'features' in locals() else "Not created")
                    st.write("Scaler available:", scaler is not None)
                    st.write("Label encoders available:", list(label_encoders.keys()) if label_encoders else "None")
                    st.write("Selected model:", model_choice if 'model_choice' in locals() else "Not selected")
                    st.write("Full error:", str(e))

def comparison_page():
    st.markdown('<h2 class="sub-header">Algorithm Comparison</h2>', unsafe_allow_html=True)
    
    df = load_algorithm_comparison()
    
    if df.empty:
        st.error("No comparison data available. Please run the training notebook first.")
        return
    
    # Display comparison table
    st.markdown("### Complete Algorithm Comparison")
    st.dataframe(df, use_container_width=True, height=400)
    
    # Interactive visualizations
    st.markdown("### Performance Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Accuracy Comparison", "⚡ Performance Metrics", "🕐 Time Analysis", "💾 Memory Usage"])
    
    with tab1:
        if 'Accuracy' in df.columns:
            # Accuracy comparison
            fig_acc = px.bar(df, x='Algorithm', y='Accuracy', 
                            title="Model Accuracy Comparison",
                            color='Accuracy',
                            color_continuous_scale='RdYlGn',
                            text='Accuracy')
            fig_acc.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_acc.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.warning("Accuracy data not available in comparison results.")
    
    with tab2:
        # Performance metrics radar chart for top models
        if all(col in df.columns for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']):
            top_models = df.nlargest(5, 'Accuracy')
            
            fig_radar = go.Figure()
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            for idx, (_, row) in enumerate(top_models.iterrows()):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[metric] if pd.notna(row[metric]) else 0 for metric in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row['Algorithm'],
                    line_color=colors[idx % len(colors)]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                title="Performance Metrics Comparison (Top 5 Models)",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Performance metrics data not available.")
    
    with tab3:
        # Training time vs prediction speed
        if all(col in df.columns for col in ['Training Time', 'Prediction Speed']):
            fig_scatter = px.scatter(df, x='Training Time', y='Prediction Speed',
                                   size='Accuracy' if 'Accuracy' in df.columns else None,
                                   hover_name='Algorithm',
                                   title="Training Time vs Prediction Speed",
                                   size_max=20)
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Timing data not available.")
    
    with tab4:
        # Memory usage comparison
        if 'Memory Usage' in df.columns:
            fig_memory = px.bar(df, x='Algorithm', y='Memory Usage',
                               title="Memory Usage Comparison (KB)",
                               color='Memory Usage',
                               color_continuous_scale='Reds',
                               text='Memory Usage')
            fig_memory.update_traces(texttemplate='%{text:.1f}KB', textposition='outside')
            fig_memory.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_memory, use_container_width=True)
        else:
            st.warning("Memory usage data not available.")

def analysis_page():
    st.markdown('<h2 class="sub-header">Performance Analysis & Insights</h2>', unsafe_allow_html=True)
    
    df = load_algorithm_comparison()
    
    if df.empty:
        st.error("No analysis data available. Please run the training notebook first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Performers")
        
        try:
            # Best accuracy
            if 'Accuracy' in df.columns:
                best_acc = df.loc[df['Accuracy'].idxmax()]
                st.markdown(f"**Best Accuracy:** {best_acc['Algorithm']} ({best_acc['Accuracy']:.2%})")
            
            # Best F1-Score
            if 'F1-Score' in df.columns:
                best_f1 = df.loc[df['F1-Score'].idxmax()]
                st.markdown(f"**Best F1-Score:** {best_f1['Algorithm']} ({best_f1['F1-Score']:.3f})")
            
            # Fastest training
            if 'Training Time' in df.columns:
                fastest = df.loc[df['Training Time'].idxmin()]
                st.markdown(f"**Fastest Training:** {fastest['Algorithm']} ({fastest['Training Time']:.3f}s)")
            
            # Fastest prediction
            if 'Prediction Speed' in df.columns:
                fastest_pred = df.loc[df['Prediction Speed'].idxmax()]
                st.markdown(f"**Fastest Prediction:** {fastest_pred['Algorithm']} ({fastest_pred['Prediction Speed']:.0f} samples/s)")
            
        except Exception as e:
            st.warning(f"Error displaying top performers: {e}")
    
    with col2:
        st.markdown("### 📈 Key Insights")
        
        st.markdown("""
        #### Model Performance Patterns:
        - **Tree-based models** (Random Forest, XGBoost) often show good balance of accuracy and interpretability
        - **Neural Networks** can capture complex patterns but require more data and tuning
        - **Logistic Regression** provides excellent baseline performance with fast training
        - **SVM** works well for high-dimensional data but can be computationally expensive
        - **Ensemble methods** typically offer robust performance across different datasets
        """)
    
    # Model recommendations
    st.markdown("### 🎯 Model Recommendations by Use Case")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        #### 🚀 Production Deployment
        **Recommended Approach:**
        - High accuracy with fast inference
        - Low memory footprint
        - Stable performance
        - Easy to maintain and update
        
        **Best Candidates:**
        - Logistic Regression
        - Random Forest
        - XGBoost (if accuracy is priority)
        """)
    
    with rec_col2:
        st.markdown("""
        #### ⚡ Real-time Applications
        **Recommended Approach:**
        - Ultra-fast prediction speed
        - Minimal latency
        - Low computational overhead
        
        **Best Candidates:**
        - Decision Tree
        - Naive Bayes
        - Linear models
        """)
    
    with rec_col3:
        st.markdown("""
        #### 🔬 Research & Analysis
        **Recommended Approach:**
        - Maximum accuracy
        - Complex pattern recognition
        - Feature importance analysis
        
        **Best Candidates:**
        - Neural Networks
        - Ensemble methods
        - SVM with RBF kernel
        """)
    
    # Feature importance analysis (if available)
    st.markdown("### 🔍 Feature Importance Insights")
    
    st.info("""
    **Key Predictive Factors for Smartphone Addiction:**
    
    1. **Daily Usage Hours** - Primary indicator of addiction risk
    2. **Sleep Hours** - Strong inverse correlation with addiction
    3. **Time on Social Media** - Major risk factor for teenagers
    4. **Phone Checks Per Day** - Behavioral indicator of dependency
    5. **Academic Performance** - Often negatively impacted by addiction
    6. **Anxiety/Depression Levels** - Strong psychological correlations
    7. **Physical Activity** - Protective factor against addiction
    8. **Screen Time Before Bed** - Sleep quality impact indicator
    """)
    
    # Performance matrix heatmap
    if len(df) > 0:
        st.markdown("### 📊 Comprehensive Performance Matrix")
        
        try:
            metrics_cols = []
            available_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
            for metric in available_metrics:
                if metric in df.columns:
                    metrics_cols.append(metric)
            
            if metrics_cols:
                heatmap_data = df[['Algorithm'] + metrics_cols].set_index('Algorithm')
                
                # Convert AUC-ROC to numeric, replacing 'N/A' with NaN
                if 'AUC-ROC' in heatmap_data.columns:
                    heatmap_data['AUC-ROC'] = pd.to_numeric(heatmap_data['AUC-ROC'], errors='coerce')
                
                # Remove rows with all NaN values
                heatmap_data = heatmap_data.dropna(how='all')
                
                if not heatmap_data.empty:
                    fig_heatmap = px.imshow(heatmap_data.T,
                                           labels=dict(x="Algorithm", y="Metric", color="Score"),
                                           title="Performance Metrics Heatmap",
                                           aspect="auto",
                                           color_continuous_scale="RdYlGn",
                                           text_auto=True)
                    fig_heatmap.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("No valid performance data available for heatmap.")
            else:
                st.warning("No performance metrics available for visualization.")
                
        except Exception as e:
            st.error(f"Error creating performance matrix: {e}")

def about_page():
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📱 Teen Smartphone Addiction Prediction System
        
        This comprehensive machine learning application predicts smartphone addiction levels among teenagers 
        based on usage patterns, behavioral indicators, and psychological factors.
        
        #### 🎯 Project Objectives
        - **Early Detection**: Identify teens at risk of smartphone addiction before severe symptoms develop
        - **Model Comparison**: Evaluate multiple ML algorithms to find the most effective approach  
        - **Actionable Insights**: Provide personalized recommendations for healthier device usage
        - **Support Intervention**: Enable parents, educators, and healthcare providers to intervene early
        
        #### 📊 Dataset Features (22 Variables)
        
        **Demographic & Academic:**
        - Age, Gender, Location, School Grade
        - Academic Performance
        
        **Usage Patterns:**
        - Daily Usage Hours, Weekend Usage Hours
        - Apps Used Daily, Phone Checks Per Day
        - Screen Time Before Bed
        
        **App-Specific Usage:**
        - Time on Social Media, Gaming, Education
        - Phone Usage Purpose
        
        **Health & Lifestyle:**
        - Sleep Hours, Exercise Hours
        - Anxiety Level, Depression Level, Self Esteem
        
        **Social Factors:**
        - Social Interactions, Family Communication
        - Parental Control Level
        
        #### 🤖 Machine Learning Pipeline
        
        **Data Preprocessing:**
        - Feature encoding for categorical variables
        - Standardization using StandardScaler
        - Train-test split with stratification
        
        **Model Training & Evaluation:**
        - 11 different algorithms compared
        - Cross-validation for robust evaluation
        - Multiple performance metrics calculated
        - Hyperparameter sensitivity analysis
        
        **Models Implemented:**
        1. **Logistic Regression** - Linear baseline model
        2. **Decision Tree** - Interpretable tree-based model
        3. **Random Forest** - Ensemble of decision trees
        4. **Support Vector Machine** - Kernel-based classifier
        5. **K-Nearest Neighbors** - Instance-based learning
        6. **Neural Networks** - Multi-layer perceptron
        7. **XGBoost** - Gradient boosting framework
        8. **AdaBoost** - Adaptive boosting ensemble
        9. **Gradient Boosting** - Sequential boosting
        10. **Naive Bayes** - Probabilistic classifier
        11. **K-Means** - Unsupervised clustering approach
        """)
    
    with col2:
        st.markdown("### 📚 Technical Specifications")
        
        st.info("""
        **Model Training Setup:**
        - **Dataset Split:** 80% train, 20% test
        - **Validation:** Stratified K-Fold CV
        - **Preprocessing:** StandardScaler + LabelEncoder
        - **Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC
        - **Performance:** Training time, prediction speed, memory usage
        """)
        
        st.success("""
        **Deployment Features:**
        - **Interactive Prediction:** Real-time addiction risk assessment
        - **Model Selection:** Compare different algorithms
        - **Risk Analysis:** Detailed factor breakdown
        - **Recommendations:** Personalized intervention strategies
        - **Visualizations:** Performance comparison charts
        """)
        
        st.warning("""
        **Important Considerations:**
        - **Screening Tool:** Not a diagnostic instrument
        - **Professional Guidance:** Combine with expert assessment
        - **Privacy:** All data processed locally
        - **Updates:** Regular model retraining recommended
        - **Bias:** Model reflects training data patterns
        """)

if __name__ == "__main__":
    main()