🔮 Odyssey | Strategic Command Center
🚀 Overview
Welcome to Odyssey, an enterprise-grade analytics dashboard designed for the strategic review of the Global Superstore dataset. This is more than a report; it's an interactive command center that transforms raw data into actionable intelligence.

Built with Python, Streamlit, and Plotly, this tool leverages AI-powered forecasting and interactive scenario planning to empower stakeholders with the ability to explore trends, identify opportunities, and make data-driven decisions with confidence.

✨ Live Demo
The application is deployed and live on Streamlit Community Cloud.

Click the badge below or this link to access the live dashboard:

https://odyssey-superstore.streamlit.app/

✨ Key Features
AI Executive Summary: A dynamic, natural-language summary that provides an immediate, intelligent overview of the current data view.

AI-Powered Forecasting: Utilizes the Prophet model to generate 90-day sales forecasts, complete with confidence intervals and anomaly detection, allowing for proactive planning.

Interactive "What-If" Scenario Planner: A powerful simulation tool to model the financial impact of strategic decisions, such as adjusting discount rates, sales targets, and operational costs.

Advanced 3D Visualization: An interactive 3D scatter plot provides a multi-dimensional view of the relationship between Sales, Profit, and Discount across product sub-categories.

Revenue Flow Analysis: A Sankey diagram visually decomposes how revenue flows from global markets through to specific product categories.

Robust Filtering & Data Export: Granular controls for date range, market, and category, with the ability to export the filtered dataset to CSV.

Polished, Professional UI: A custom-themed "Command Center" interface designed for clarity, impact, and a premium user experience.

🖥️ Dashboard Preview
(Replace this with a new screenshot of your running "Odyssey" dashboard)


🛠️ How to Run Locally
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8 or higher

Pip package manager

1. Clone the Repository
git clone https://github.com/ms347135/odyssey-superstore-dashboard.git
cd odyssey-superstore-dashboard

2. Set Up a Virtual Environment (Recommended)
Creating a virtual environment is a best practice for managing project dependencies cleanly.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
This project uses advanced libraries, including prophet for forecasting. Install all dependencies from the requirements.txt file.

pip install -r requirements.txt

(Note: prophet can sometimes have complex dependencies. If you encounter issues, refer to the official Prophet installation guide.)

4. Run the Streamlit Application
Launch the interactive dashboard with the following command:

streamlit run dashboard.py

Your web browser should automatically open with the dashboard running, typically at http://localhost:8501.

📁 Project Structure
odyssey-superstore-dashboard/
├── .gitignore               # Files and folders to ignore for Git
├── requirements.txt         # Project dependencies (includes prophet)
├── dashboard.py             # The main Streamlit dashboard application
└── README.md                # This file

👨‍💻 Developer
This dashboard was designed and developed by Malik Saad.

Contact: ms347135@gmail.com

Feel free to reach out for collaboration or inquiries.