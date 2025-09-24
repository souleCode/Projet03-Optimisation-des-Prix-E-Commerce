
"""
Usage:
    python load_kaggle_olist.py --csv-path data/raw --pg-uri postgresql://user:pass@localhost:5432/dbname
"""

from __future__ import annotations
import os
import argparse
import logging
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------- Configuration ----------
DEFAULT_FILES: Dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "product_category_name_translation": "product_category_name_translation.csv",
}
CHUNKSIZE = 10000  # utilisé si vous voulez charger en chunk
# -----------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("olist_ingestion")


def get_engine(pg_uri: str) -> Engine:
    """Crée un engine SQLAlchemy."""
    engine = create_engine(pg_uri, pool_pre_ping=True)
    return engine


def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """Lit un CSV en gérant les erreurs courantes."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    logger.info("Lecture CSV: %s", path)
    df = pd.read_csv(path, low_memory=False, **kwargs)
    logger.info(" -> shape: %s", df.shape)
    return df


def load_df_to_postgres(df: pd.DataFrame, table_name: str, engine: Engine, if_exists: str = "replace") -> None:
    """Charge un DataFrame dans Postgres avec to_sql (chunksize pour gros volumes)."""
    logger.info("Chargement en base: %s (if_exists=%s)", table_name, if_exists)
    # essayer méthode 'multi' pour accélérer
    df.to_sql(table_name, con=engine, if_exists=if_exists, index=False, method="multi", chunksize=CHUNKSIZE)
    logger.info(" -> Chargé: %s (%d lignes)", table_name, len(df))


def create_indexes(engine: Engine) -> None:
    """Créer quelques index utiles pour les requêtes analytiques."""
    logger.info("Création des index utiles...")
    idx_statements = [
        "CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_purchase_timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_products_product_id ON products(product_id);",
        # index sur enriched table (créée ensuite)
    ]
    with engine.begin() as conn:
        for stmt in idx_statements:
            logger.debug("Executing: %s", stmt)
            conn.execute(text(stmt))
    logger.info("Indexation basique créée.")


def create_enriched_orders(engine: Engine, csv_path: str, force_rebuild: bool = True) -> None:
    """
    Construit une table 'orders_enriched' en joignant order_items + orders + products.
    Cette opération est faite en mémoire ici (dataset Olist est modestement dimensionné).
    Si vous avez trop de données, faites la jointure en SQL ou par chunks.
    """
    logger.info("Création de orders_enriched à partir des tables order_items, orders, products")
    # lire depuis Postgres si déjà chargées, sinon depuis CSV
    try:
        # prefer reading from DB (si tables présentes)
        orders = pd.read_sql_table("orders", con=engine)
        order_items = pd.read_sql_table("order_items", con=engine)
        products = pd.read_sql_table("products", con=engine)
        logger.info("Chargé depuis Postgres: orders(%s), order_items(%s), products(%s)",
                    orders.shape, order_items.shape, products.shape)
    except Exception:
        logger.info("Tables manquantes en base -> lecture CSVs depuis %s", csv_path)
        orders = read_csv_safe(os.path.join(csv_path, DEFAULT_FILES["orders"]),
                               parse_dates=["order_purchase_timestamp", "order_approved_at", "order_delivered_customer_date"],
                               infer_datetime_format=True)
        order_items = read_csv_safe(os.path.join(csv_path, DEFAULT_FILES["order_items"]))
        products = read_csv_safe(os.path.join(csv_path, DEFAULT_FILES["products"]))

    # nettoyage minimal
    logger.info("Nettoyage minimal : suppression des lignes sans price ou sans product_id")
    order_items = order_items[order_items['price'].notnull() & order_items['product_id'].notnull()]
    products = products.drop_duplicates(subset=['product_id'])

    # jointures en mémoire
    logger.info("Jointure order_items <- orders (on order_id)")
    df = order_items.merge(orders, on='order_id', how='left', suffixes=('_item', '_order'))
    logger.info(" -> shape après première jointure: %s", df.shape)

    logger.info("Jointure result <- products (on product_id)")
    df = df.merge(products, on='product_id', how='left', suffixes=('', '_product'))
    logger.info(" -> shape après jointure produits: %s", df.shape)

    # normaliser dates et types utiles
    date_cols = [c for c in df.columns if 'timestamp' in c or 'date' in c]
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            pass

    # supprimer enregistrements manifestement erronés
    df = df[df['price'].apply(lambda x: pd.notnull(x) and (float(x) >= 0))]
    logger.info(" -> shape après nettoyage final: %s", df.shape)

    # écrire en base (remplace si exists)
    load_df_to_postgres(df, "orders_enriched", engine, if_exists="replace")

    # ajouter index spécifiques à la table enrichie
    idx_statements = [
        "CREATE INDEX IF NOT EXISTS idx_orders_enriched_product_id ON orders_enriched(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_enriched_order_date ON orders_enriched(order_purchase_timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_orders_enriched_category ON orders_enriched(product_category_name);"
    ]
    with engine.begin() as conn:
        for stmt in idx_statements:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                logger.warning("Erreur création index: %s", e)

    logger.info("Table orders_enriched créée et indexées.")


def ingest_all(csv_path: str, engine: Engine, overwrite: bool = True) -> None:
    """
    Charge les CSV listés dans DEFAULT_FILES vers Postgres.
    Puis crée la table enrichie (orders_enriched).
    """
    logger.info("Start ingestion from %s", csv_path)
    for key, fname in DEFAULT_FILES.items():
        fpath = os.path.join(csv_path, fname)
        if not os.path.exists(fpath):
            logger.warning("Fichier absent (skippé): %s", fpath)
            continue
        # choix parse_dates pour certaines tables
        parse_dates = None
        if key == "orders":
            parse_dates = ["order_purchase_timestamp", "order_approved_at", "order_delivered_customer_date"]
        if key == "reviews":
            parse_dates = ["review_creation_date", "review_answer_timestamp"]
        df = read_csv_safe(fpath, parse_dates=parse_dates, infer_datetime_format=True)
        # some light cleaning
        df = df.drop_duplicates()
        table_name = key if key != "order_items" else "order_items"
        load_df_to_postgres(df, table_name, engine, if_exists="replace" if overwrite else "append")

    # créer quelques indexes de base
    create_indexes(engine)

    # créer la table enrichie
    create_enriched_orders(engine, csv_path, force_rebuild=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load Olist Kaggle CSVs to Postgres and create enriched orders table.")
    p.add_argument("--csv-path", type=str, default="../../data", help="Chemin vers le dossier contenant les CSV Olist")
    p.add_argument("--pg-uri", type=str, default=os.environ.get("POSTGRES_URI"),
                   help="URI Postgres (ex: postgresql://user:pass@host:5432/dbname) ou env POSTGRES_URI")
    p.add_argument("--no-overwrite", action="store_true", help="Ne pas remplacer les tables existantes (append)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.pg_uri:
        logger.error("Aucune PG URI fournie. Passez --pg-uri ou définissez POSTGRES_URI dans l'environnement.")
        raise SystemExit(1)

    engine = get_engine(args.pg_uri)
    ingest_all(args.csv_path, engine, overwrite=not args.no_overwrite)
    logger.info("Ingestion terminée.")


if __name__ == "__main__":
    main()
