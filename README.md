# LLM Agent for Energy Efficiency in Residential Housing

An intelligent LLM-driven agent that analyzes smart meter data to provide personalized energy insights and recommendations for residential customers. This project combines advanced data analytics with natural language processing to make energy consumption patterns accessible and actionable.

##  Features

- **Smart Meter Data Analysis**: Comprehensive analysis of residential energy consumption patterns
- **AI-Powered Insights**: Uses Anthropic's Claude LLM to generate intelligent recommendations
- **Interactive Visualizations**: Dynamic Plotly graphs showing peak hours, seasonal variations, and weekly patterns
- **Customer-Centric Interface**: User-friendly Streamlit web application for easy exploration
- **Real-time Chat**: Ask questions about energy data and get instant AI-generated insights
- **Multi-Site Analysis**: Supports analysis of 236+ residential sites across multiple states

##  Live Demo

**[Access the app here](https://llm-agentforenergyefficiencyinresdentialhousing-b5rfdgnpkizwvc.streamlit.app/)**



## Architecture

```
├── app.py                    # Main Streamlit web application 
├── Site_analysis.py         # Core analysis engine and data processing (all the files in analyzed_sites were created with this script)
├── analyzed_sites/          # Processed data for 236+ residential sites
│   ├── [site_id]/
│   │   ├── site_analysis_[id].json           # Analysis data
│   │   ├── peak_hour_load_[id].html         # Peak hours graph
│   │   ├── Seasonal_load_profile_[id].html  # Seasonal analysis
│   │   └── Week_profile_[id].html           # Weekly patterns
├── kmeans_pipeline.joblib   # ML model for customer segmentation
├── state_averages.csv       # Reference data for comparative analysis
└── requirements.txt         # Python dependencies
```

##  Installation & Setup

### Prerequisites

- Python 3.12+
- Anthropic API key for Claude LLM if running locally
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/julescrst/LLM-agent_for_energy_efficiency_in_resdential_housing.git
   cd LLM-agent_for_energy_efficiency_in_resdential_housing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```
  

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   
   Open your browser and navigate to `http://localhost:8501`

##  How to Use

### Web Application (app.py)

1. **Select a Site**: Choose from 236+ available residential sites in the sidebar
2. **View Analysis**: Get automatic AI-generated insights about the selected site
3. **Explore Graphs**: Interactive visualizations of energy patterns:
   - Peak Hour Load
   - Seasonal Variation  
   - Weekly Profile
4. **Ask Questions**: Use the chat interface to get specific insights about energy data
5. **Download Data**: Access raw JSON data for further analysis

### Data Analysis (Site_analysis.py)

Process new sites or regenerate analysis:

```bash
# Analyze a specific site
python Site_analysis.py --site 727 --state WA

# List all available sites
python Site_analysis.py --list

# Process all sites (bulk processing)
python Site_analysis.py --all
```

## 📈 Data Processing Pipeline

The analysis pipeline includes:

1. **Data Loading**: Reads 60-minute interval smart meter data
2. **Pattern Detection**: Identifies peak hours, seasonal trends, and weekly patterns and more
3. **ML Clustering**: Uses K-means to segment houses into consumption profiles
4. **Comparative Analysis**: Benchmarks against state averages
5. **Visualization Generation**: Creates interactive Plotly graphs
6. **AI Insights**: Generates personalized recommendations using Claude LLM

## 🧠 AI Features

- **Natural Language Queries**: Ask questions like "What are my peak consumption hours?"
- **Contextual Recommendations**: AI analyzes patterns and provides actionable advice
- **Comparative Insights**: Explains how consumption compares to similar households
- **Energy Efficiency Tips**: Suggests specific improvements based on usage patterns


## 📊 Sample Data

The system includes analysis for 236 residential sites with features like:
- Hourly energy consumption patterns
- Temperature correlation analysis
- Peak demand identification
- Seasonal variation tracking
- Customer segmentation profiles


## 📝 License

This project is part of an EPFL Semester Project (MA3) focused on energy efficiency analysis.


### Cloud Deployment

The app is optimized for Streamlit Cloud deployment with proper secret management


**Built with intention to improve sustainable energy management**