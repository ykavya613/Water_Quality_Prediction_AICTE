import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="💧 Water Quality Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling (Fresh & Clean Theme with requested adjustments) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #191970; /* Dark Blue for all general text */
    }

    .main {
        /* Pure white background for the entire page */
        background-image: url('https://placehold.co/1920x1080/FFFFFF/FFFFFF'); /* Pure white background */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .stApp {
        /* Very light blue background for the main content area with high opacity */
        background-color: rgba(240, 248, 255, 0.98); /* AliceBlue */
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #008080; /* Teal for headings */
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .stButton>button {
        background-color: #66CDAA; /* Soft Green button (MediumAquamarine) */
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    .stButton>button:hover {
        background-color: #45B39D; /* Slightly darker green on hover */
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #ADD8E6; /* Light blue border for dataframes */
        background-color: rgba(255, 165, 0, 0.8); /* Opaque Orange background for dataframes */
        color: #000000; /* Black text for dataframe content */
    }

    /* Ensure text within dataframe cells is black */
    .stDataFrame table th, .stDataFrame table td {
        color: #000000 !important;
    }


    .stNumberInput, .stSelectbox {
        background-color: #3CB371; /* Medium Sea Green for input fields background */
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid #2E8B57; /* Sea Green border */
        color: #FFFFFF; /* White text for input labels */
    }

    /* Ensure the text *inside* the number input is white */
    .stNumberInput input {
        color: #FFFFFF !important;
    }

    .stAlert {
        border-radius: 8px;
        /* Streamlit's default alert colors are usually distinct, keeping them */
    }

    .sidebar .sidebar-content {
        background-color: #E0FFFF; /* Lighter Teal for sidebar content (LightCyan) */
        border-right: 2px solid #008080; /* Teal for sidebar border */
        padding: 1rem;
        border-radius: 0 15px 15px 0;
        color: #FFFFFF; /* White text for sidebar content */
    }

    /* Ensure headings inside sidebar are also teal */
    .sidebar h1, .sidebar h2, .sidebar h3, .sidebar h4, .sidebar h5, .sidebar h6 {
        color: #008080; /* Headings in sidebar are teal, as per Option 1 */
    }

    /* Explicitly set markdown text in sidebar to white */
    .sidebar .stMarkdown {
        color: #FFFFFF !important;
    }

    .stMarkdown {
        line-height: 1.6;
    }

    /* Ensure Plotly charts remain colorful */
    .stPlotlyChart {
        background-color: transparent !important; /* Ensure chart background is transparent */
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("💧 Water Quality Pollutants Predictor 🌊") # Added water emoji
st.markdown("### Intelligent Forecasting for Clean Water Monitoring")

# --- Load dataset from local file ---
@st.cache_data
def load_data():
    """
    Loads the water quality dataset from a CSV file.
    Caches the data to improve performance on subsequent runs.
    """
    try:
        df = pd.read_csv("PB_All_2000_2021.csv", sep=';')
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        return df
    except FileNotFoundError:
        st.error("Error: 'PB_All_2000_2021.csv' not found. Please ensure the dataset is in the same directory as app.py.")
        st.stop() # Stop the app if the file is not found
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

df = load_data()

# Define the pollutants we are interested in
pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Drop rows where all pollutant values are missing
df = df.dropna(subset=pollutants, how='all')

# --- Data Preprocessing and Model Training ---
# Feature/Target split
X = df[['id', 'year', 'month']]
y = df[pollutants]

# Impute missing values in the target variables using the mean strategy
imputer = SimpleImputer(strategy='mean')
y_imputed = imputer.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_imputed, test_size=0.2, random_state=42)

# Model: MultiOutputRegressor with RandomForestRegressor
# RandomForestRegressor is chosen for its robustness and ability to handle non-linear relationships
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
model = MultiOutputRegressor(rf)

# Train the model
with st.spinner("Training the prediction model... This might take a moment."):
    model.fit(X_train, y_train)
st.success("Model training complete! 🎉") # Added emoji

# Make predictions on the test set
y_pred = model.predict(X_test)

# --- Model Evaluation ---
st.subheader("📊 Model Performance Overview")
st.markdown("Here's how well our model performed on unseen data:")

# Calculate R² Score for each pollutant
r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
# Calculate RMSE for each pollutant
rmse_scores = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

# Create a DataFrame to display results
results = pd.DataFrame({
    'Pollutant': pollutants,
    'R² Score': r2_scores,
    'RMSE': rmse_scores
}).sort_values(by='R² Score', ascending=False)

st.dataframe(results.style.format({'R² Score': "{:.3f}", 'RMSE': "{:.2f}"}))

st.markdown("""
    * **R² Score**: Indicates the proportion of variance in the dependent variable that can be predicted from the independent variables. A higher R² (closer to 1.0) is better.
    * **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors. A lower RMSE is better.
""")

# --- R² Score Visualization ---
st.subheader("📈 Prediction Accuracy (R² Score) by Pollutant")
st.markdown("A higher R² score indicates better prediction accuracy for that pollutant.")

fig_r2 = px.bar(
    results,
    x='Pollutant',
    y='R² Score',
    title='R² Score for Each Pollutant',
    labels={'R² Score': 'R² Score'},
    color='R² Score', # Color bars based on R² score
    color_continuous_scale=px.colors.sequential.Viridis, # Choose a colorful scale for the graph
    hover_data={'Pollutant': True, 'R² Score': ':.3f', 'RMSE': ':.2f'}
)

fig_r2.update_layout(
    xaxis_title="Pollutant Type",
    yaxis_title="R² Score",
    plot_bgcolor='rgba(0,0,0,0)', # Transparent background for the plot
    paper_bgcolor='rgba(0,0,0,0)', # Transparent background for the paper
    margin=dict(l=50, r=50, t=80, b=50),
    hovermode="x unified",
    font_color="black", # Set general font color for the plot to black
    xaxis_title_font_color="black", # Set x-axis title font color to black
    yaxis_title_font_color="black" # Set y-axis title font color to black
)
st.plotly_chart(fig_r2, use_container_width=True)


# --- Prediction Section ---
st.subheader("🔍 Predict Water Quality for a Specific Station 🧪") # Added emoji
st.markdown("Enter the details below to get a prediction for pollutant levels.")

# Create columns for better layout of input fields
col1, col2, col3 = st.columns(3)

with col1:
    id_input = st.number_input("Station ID", min_value=df['id'].min(), max_value=df['id'].max(), value=int(df['id'].mean()), help="Enter the ID of the water quality monitoring station.")
with col2:
    year_input = st.number_input("Year", min_value=2000, max_value=datetime.datetime.now().year + 5, value=datetime.datetime.now().year, help="Enter the year for prediction.")
with col3:
    month_input = st.number_input("Month", min_value=1, max_value=12, value=datetime.datetime.now().month, help="Enter the month (1-12) for prediction.")

if st.button("Predict Pollutant Levels"):
    # Prepare input for prediction
    input_df = pd.DataFrame([[id_input, year_input, month_input]], columns=['id', 'year', 'month'])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Create a DataFrame for predicted values
    pred_df = pd.DataFrame({'Pollutant': pollutants, 'Predicted Value': prediction})

    st.subheader("🔮 Predicted Pollutant Levels ✨") # Added emoji
    st.dataframe(pred_df.style.format({'Predicted Value': "{:.3f}"}))

    # --- Interactive Graph for Pollutant Levels ---
    st.subheader("📈 Predicted Pollutant Levels Visualization 📉") # Added emoji
    st.markdown("This chart shows the predicted levels for each pollutant. Values exceeding typical safe levels are highlighted.")

    # Define "safe" thresholds (these are example values, you should replace them with actual standards)
    # These thresholds are crucial for the "dangerous" depiction
    thresholds = {
        'NH4': 0.5,       # Ammonia
        'BSK5': 5.0,      # BOD5 (Biochemical Oxygen Demand)
        'Suspended': 25.0, # Suspended Solids
        'O2': 7.0,        # Dissolved Oxygen (higher is generally better, so lower than this is 'dangerous')
        'NO3': 10.0,      # Nitrates
        'NO2': 1.0,       # Nitrites
        'SO4': 250.0,     # Sulfates
        'PO4': 0.1,       # Phosphates
        'CL': 250.0       # Chlorides
    }

    # Add a 'Status' column based on thresholds
    # For O2, lower values are dangerous. For others, higher values are dangerous.
    pred_df['Status'] = pred_df.apply(
        lambda row: 'Dangerous' if (row['Pollutant'] == 'O2' and row['Predicted Value'] < thresholds.get(row['Pollutant'], 0)) or \
                                     (row['Pollutant'] != 'O2' and row['Predicted Value'] > thresholds.get(row['Pollutant'], 0))
                                  else 'Safe',
        axis=1
    )

    # Create a bar chart using Plotly Express
    fig = px.bar(
        pred_df,
        x='Pollutant',
        y='Predicted Value',
        color='Status',
        color_discrete_map={'Safe': '#28a745', 'Dangerous': '#dc3545'}, # Green for safe, red for dangerous (these colors are kept for the graph)
        title='Predicted Pollutant Levels and Safety Status',
        labels={'Predicted Value': 'Concentration (Units)'},
        hover_data={'Pollutant': True, 'Predicted Value': ':.3f', 'Status': True}
    )

    # Add a line for the threshold for each pollutant (if applicable and useful)
    for pollutant, threshold in thresholds.items():
        if pollutant in pred_df['Pollutant'].values:
            # Get the predicted value for the current pollutant
            predicted_value = pred_df[pred_df['Pollutant'] == pollutant]['Predicted Value'].iloc[0]

            # Add a horizontal line for the threshold
            fig.add_shape(
                type="line",
                x0=pollutant, y0=threshold, x1=pollutant, y1=threshold,
                line=dict(color="blue", width=2, dash="dash"),
                name=f'{pollutant} Threshold'
            )
            # Add text annotation for the threshold
            fig.add_annotation(
                x=pollutant,
                y=threshold,
                text=f'Threshold: {threshold}',
                showarrow=False,
                yshift=10,
                font=dict(color="blue", size=10)
            )

    fig.update_layout(
        xaxis_title="Pollutant Type",
        yaxis_title="Predicted Concentration",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        font_color="black", # Set general font color for the plot to black
        xaxis_title_font_color="black", # Set x-axis title font color to black
        yaxis_title_font_color="black" # Set y-axis title font color to black
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Note on Safety Levels**: The 'Dangerous' status is determined by predefined thresholds for each pollutant.
    These thresholds are illustrative and should be replaced with official water quality standards relevant to your region.
    For **Dissolved Oxygen (O2)**, a lower value indicates danger, while for other pollutants, a higher value indicates danger.
    """)

# --- Project Details and Dataset Sections in Sidebar (Removed as per request) ---
# st.sidebar.subheader("Project Details")
# st.sidebar.markdown(...)
# st.sidebar.subheader("Dataset")
# st.sidebar.markdown(...)
