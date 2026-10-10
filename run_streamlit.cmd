@echo off
cd /d "%~dp0"
set PORT=8502
echo Starting final Streamlit dashboard from app.py on port %PORT%...
.venv\Scripts\python.exe -m streamlit run app.py --server.port %PORT%
