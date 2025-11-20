import os
import dotenv
import streamlit as st

dotenv.load_dotenv()

import streamlit as st
import json
import os
from pathlib import Path
import anthropic
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Energy Usage Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_json_data' not in st.session_state:
    st.session_state.current_json_data = None
if 'current_site_id' not in st.session_state:
    st.session_state.current_site_id = None
if 'initial_analysis_pending' not in st.session_state:
    st.session_state.initial_analysis_pending = False

# Configuration
JSON_DATA_PATH = "processed_JSON"  

ANTHROPIC_API_KEY  = st.secrets["ANTHROPIC_API_KEY"] if "ANTHROPIC_API_KEY" in st.secrets else os.getenv("ANTHROPIC_API_KEY")

# Helper Functions
@st.cache_data
def load_json_file(site_id):
    """Load JSON file for a given site ID"""
    try:
        
        filename = f"site_analysis_{site_id}.json"
        
        filepath = Path(JSON_DATA_PATH) / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_available_sites():
    """Get list of available site IDs from JSON directory"""
    try:
        json_path = Path(JSON_DATA_PATH)
        if not json_path.exists():
            return []
        
        files = list(json_path.glob("*.json"))
        site_ids = []
        for file in files:
            # Extract site ID from filename
            name = file.stem
            site_id = name.replace("site_", "").replace("_", "")
            site_ids.append(site_id)
        return sorted(site_ids)
    except:
        return []

def query_claude(json_data, user_question, site_id):
    """Query Claude API with JSON data and user question"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Create the prompt
        prompt = f"""You are an expert energy usage analyst. You have been provided with detailed energy consumption data for household site {site_id}.

Below is the complete energy usage data in JSON format:

{json.dumps(json_data, indent=2)}

User Question: {user_question}

Please provide a clear, insightful analysis that:
1. Directly answers the user's question
2. Highlights key patterns, trends, or anomalies in the data
3. Offers actionable recommendations where relevant
4. Uses specific numbers and data points from the JSON
5. Keeps the response concise but comprehensive

Response:"""

        # Stream the response
        response_container = st.empty()
        full_response = ""

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"Error querying Claude API: {str(e)}")
        return None



# Main App
def main():
    # Header
    st.title("🏠 Household Energy Usage Analyzer")
    st.markdown("*Analyze household energy consumption patterns with AI-powered insights*")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input if not set in secrets
        if not ANTHROPIC_API_KEY:
            api_key_input = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Enter your Claude API key"
            )
            if api_key_input:
                globals()['ANTHROPIC_API_KEY'] = api_key_input
        else:
            st.success("✓ API Key Configured")
        
        st.divider()
        
        # Site ID Selection
        st.subheader("Site Selection")
        
        
        
        # Site ID input
        site_id = st.text_input(
            "Enter Site ID",
            value="",
            placeholder="e.g., 001 or site_001",
            help="Enter the site ID to analyze"
        )
        
        load_button = st.button("🔍 Load Site Data", type="primary", use_container_width=True)
        
        # Load data
        if load_button and site_id:
            with st.spinner("Loading site data..."):
                json_data = load_json_file(site_id)
                
                if json_data:
                    st.session_state.current_json_data = json_data
                    st.session_state.current_site_id = site_id
                    st.session_state.conversation_history = []
                    st.success(f"✓ Site {site_id} loaded successfully!")
                    
                    # Generate initial analysis automatically
                    if ANTHROPIC_API_KEY:
                        st.session_state.initial_analysis_pending = True
                    
                    st.rerun()
                else:
                    st.error(f"❌ Site {site_id} not found. Please check the site ID.")
        
        # Show current loaded site
        if st.session_state.current_site_id:
            st.divider()
            st.info(f"**Current Site:** {st.session_state.current_site_id}")
            if st.button("Clear Site", use_container_width=True):
                st.session_state.current_json_data = None
                st.session_state.current_site_id = None
                st.session_state.conversation_history = []
                st.session_state.initial_analysis_pending = False
                st.rerun()
    
    # Main Content
    if st.session_state.current_json_data is None:
        # Welcome screen
        st.info("👈 Please enter a Site ID in the sidebar to begin analysis")
        
    
    else:
        # Display data for loaded site
        data = st.session_state.current_json_data
        
        # Create tabs
        tab1, tab2 = st.tabs(["🤖 AI Analysis", "💾 Raw JSON"])
        
        with tab1:
            st.subheader(f"AI-Powered Analysis for Site {st.session_state.current_site_id}")
            
            # Display summary cards
            #display_summary_cards(data)
            st.divider()
            
            # Generate initial analysis if pending
            if st.session_state.initial_analysis_pending:
                if ANTHROPIC_API_KEY:
                    st.session_state.initial_analysis_pending = False
                    
                    # Create automatic initial analysis
                    initial_question = "Please provide a comprehensive analysis of this household's energy usage data, including key insights, patterns, anomalies, and actionable recommendations to improve energy efficiency and reduce costs."
                    
                    # Add to conversation history
                    #st.session_state.conversation_history.append({
                    #    "role": "user",
                    #    "content": "📊 Initial Analysis Request"
                    #})
                    
                    # Display the automatic analysis
                    with st.chat_message("user"):
                        st.markdown("📊 **Initial Analysis Request**")
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing energy data..."):
                            response = query_claude(data, initial_question, st.session_state.current_site_id)
                            
            
            # Chat interface
            st.markdown("### Ask Questions About This Site's Energy Usage")
            
            # Display conversation history
            for message in st.session_state.conversation_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # User input
            user_question = st.chat_input("Ask a question about the energy data...")
            
            if user_question:
                if not ANTHROPIC_API_KEY:
                    st.error("Please provide an Anthropic API key in the sidebar.")
                else:
                    # Add user message to history
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_question
                    })
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(user_question)
                    
                    # Get AI response
                    with st.chat_message("assistant"):
                        response = query_claude(data, user_question, st.session_state.current_site_id)
                        
                        if response:
                            # Add assistant response to history
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": response
                            })
            
            # Suggested questions
            if not st.session_state.conversation_history:
                st.markdown("#### 💡 Suggested Questions:")
                cols = st.columns(2)
                suggestions = [
                    "What are the key insights from this household's energy usage?",
                    "When does this household consume the most energy?",
                    "What are some recommendations to reduce energy costs?",
                    "Are there any unusual patterns or anomalies?"
                ]
                
                for idx, suggestion in enumerate(suggestions):
                    col = cols[idx % 2]
                    with col:
                        if st.button(suggestion, key=f"suggestion_{idx}"):
                            st.session_state.conversation_history.append({
                                "role": "user",
                                "content": suggestion
                            })
                            st.rerun()
        
        
        
        with tab2:
            st.subheader("Raw JSON Data")
            
            # Pretty print JSON
            st.json(data)
            
            # Download button
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"site_{st.session_state.current_site_id}_data.json",
                mime="application/json"
            )
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
