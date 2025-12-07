# LLM-agent_for_energy_efficiency_in_resdential_housing
An LLM-Driven Agent for Smart Meter Data Analysis and Customer-Centric Energy Insights


[You can access the app via this link](https://llm-agentforenergyefficiencyinresdentialhousing-b5rfdgnpkizwvc.streamlit.app/)


for state in ID MT OR WA; do
  for site in residential/processed/$state/*/; do
    site_id=$(basename "$site")
    echo "Processing $state/$site_id..."
    python Site_analysis.py --site "$site_id" --state "$state"
  done
done