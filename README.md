# Multi-Agent-startup-Platform

## 1. install the following dependencies: (copy and paste)
pip install -r requirements.txt

## V1 & V2 Create a .env file to store your api keys in the following format:
OPENAI_API_KEY="sk-proj...."
TAVILY_API_KEY="tvly-dev...."

# FROM V3+
ls .streamlit
find secrets.toml

## TO START STREAMLIT
streamlit run streamlit_app.py

python knowledge_system/scripts/data_ingestion.py

python knowledge_system/scripts/setup_knowledge_bases.py