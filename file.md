ecommerce-pricing/
├─ data/                  # CSVs bruts (gitignored) et samples
├─ src/
│  ├─ ingestion/
│  │  └─ load_kaggle_olist.py
│  ├─ etl/
│  │  └─ transform.py
│  ├─ eda/
│  │  └─ eda_notebook.ipynb
│  ├─ modeling/
│  │  ├─ demand_model.py
│  │  └─ elasticity.py
│  ├─ optimization/
│  │  └─ price_optimizer.py
│  ├─ dashboard/
│  │  └─ app_streamlit.py
│  └─ ab_test/
│     └─ simulate_ab.py
├─ notebooks/
├─ requirements.txt
├─ README.md
└─ docker-compose.yml
