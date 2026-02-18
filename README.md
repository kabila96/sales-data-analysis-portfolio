# Retail Sales & Store Performance (Portfolio)

**Author:** Powell Andile Ndlovu

This portfolio includes:
- A PDF report with key findings and visuals
- A Streamlit dashboard for interactive exploration

## Files
- `Powell_Ndlovu_Sales_Portfolio_Report.pdf` (report)
- `merged_train_store_open_days.csv` (prepared dataset used by the app)
- `streamlit_dashboard_app.py` (dashboard)
- `requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard_app.py
```

## Deploy on Streamlit Cloud
1. Push the contents of this folder to a GitHub repo
2. In Streamlit Cloud, choose:
   - App file: `streamlit_dashboard_app.py`
3. (Optional) enable forecasting by uncommenting `prophet` in `requirements.txt`
