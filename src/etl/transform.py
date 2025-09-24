"""
Transformations des données Olist pour l'analyse de prix.
Crée les tables features nécessaires au modeling.

Usage:
    python transform.py --pg-uri postgresql://user:pass@localhost:5432/dbname
"""

from __future__ import annotations
import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("olist_transform")


def get_engine(pg_uri: str) -> Engine:
    """Crée un engine SQLAlchemy."""
    return create_engine(pg_uri, pool_pre_ping=True)


def create_product_features(engine: Engine) -> None:
    """
    Crée une table de features au niveau produit.
    - Prix moyen, médian, écart-type
    - Volume de ventes
    - Saisonnalité
    - Catégorie
    - Performance temporelle
    """
    logger.info("Création de la table product_features...")
    
    query = """
    WITH product_stats AS (
        SELECT 
            product_id,
            product_category_name,
            COUNT(*) as total_orders,
            AVG(price) as avg_price,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
            STDDEV(price) as price_std,
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(freight_value) as avg_freight,
            SUM(price) as total_revenue,
            COUNT(DISTINCT order_id) as unique_orders,
            COUNT(DISTINCT seller_id) as unique_sellers
        FROM orders_enriched
        WHERE price > 0 AND price IS NOT NULL
        GROUP BY product_id, product_category_name
    ),
    recency_features AS (
        SELECT
            product_id,
            MAX(order_purchase_timestamp) as last_sale_date,
            COUNT(CASE WHEN order_purchase_timestamp >= (SELECT MAX(order_purchase_timestamp) FROM orders_enriched) - INTERVAL '30 days' 
                      THEN 1 END) as sales_last_30d,
            COUNT(CASE WHEN order_purchase_timestamp >= (SELECT MAX(order_purchase_timestamp) FROM orders_enriched) - INTERVAL '90 days' 
                      THEN 1 END) as sales_last_90d
        FROM orders_enriched
        GROUP BY product_id
    )
    SELECT 
        ps.*,
        rf.last_sale_date,
        rf.sales_last_30d,
        rf.sales_last_90d,
        (rf.sales_last_30d * 1.0 / NULLIF(ps.total_orders, 0)) as sales_growth_ratio,
        CASE 
            WHEN rf.sales_last_30d > 0 THEN 'ACTIVE'
            WHEN rf.sales_last_90d > 0 THEN 'DORMANT'
            ELSE 'INACTIVE'
        END as product_status,
        -- Features de prix relatif
        (ps.avg_price / NULLIF(ps.median_price, 0)) as price_variability_ratio,
        (ps.max_price - ps.min_price) as price_range
    FROM product_stats ps
    LEFT JOIN recency_features rf ON ps.product_id = rf.product_id
    """
    
    df = pd.read_sql(query, engine)
    
    # Nettoyage supplémentaire
    df = df.fillna({
        'price_std': 0,
        'avg_freight': 0,
        'sales_last_30d': 0,
        'sales_last_90d': 0,
        'sales_growth_ratio': 0
    })
    
    # Calculer des features supplémentaires
    df['price_to_freight_ratio'] = df['avg_price'] / np.where(df['avg_freight'] > 0, df['avg_freight'], 1)
    df['revenue_per_order'] = df['total_revenue'] / df['unique_orders']
    
    # Sauvegarder en base
    df.to_sql('product_features', engine, if_exists='replace', index=False)
    
    # Créer des index
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_product_features_id ON product_features(product_id);
            CREATE INDEX IF NOT EXISTS idx_product_features_category ON product_features(product_category_name);
            CREATE INDEX IF NOT EXISTS idx_product_features_status ON product_features(product_status);
        """))
    
    logger.info("Table product_features créée avec %d produits", len(df))


def create_category_features(engine: Engine) -> None:
    """
    Crée des features au niveau catégorie de produits.
    - Performance de la catégorie
    - Prix moyens par catégorie
    - Saisonnalité par catégorie
    """
    logger.info("Création de la table category_features...")
    
    query = """
    WITH category_stats AS (
        SELECT 
            product_category_name,
            COUNT(DISTINCT product_id) as unique_products,
            COUNT(*) as total_orders,
            AVG(price) as avg_category_price,
            STDDEV(price) as price_std_category,
            SUM(price) as total_category_revenue,
            COUNT(DISTINCT seller_id) as unique_sellers_category
        FROM orders_enriched
        WHERE product_category_name IS NOT NULL
        GROUP BY product_category_name
    ),
    category_growth AS (
        SELECT
            product_category_name,
            COUNT(CASE WHEN order_purchase_timestamp >= (SELECT MAX(order_purchase_timestamp) FROM orders_enriched) - INTERVAL '30 days' 
                      THEN 1 END) as recent_orders,
            COUNT(CASE WHEN order_purchase_timestamp >= (SELECT MAX(order_purchase_timestamp) FROM orders_enriched) - INTERVAL '90 days' 
                      THEN 1 END) as last_quarter_orders
        FROM orders_enriched
        GROUP BY product_category_name
    )
    SELECT 
        cs.*,
        cg.recent_orders,
        cg.last_quarter_orders,
        (cg.recent_orders * 1.0 / NULLIF(cs.total_orders, 0)) as category_growth_rate,
        cs.total_category_revenue / NULLIF(cs.unique_products, 1) as revenue_per_product
    FROM category_stats cs
    LEFT JOIN category_growth cg ON cs.product_category_name = cg.product_category_name
    WHERE cs.total_orders >= 10  -- Filtrer les catégories trop petites
    """
    
    df = pd.read_sql(query, engine)
    
    # Calculer le market share relatif
    total_revenue = df['total_category_revenue'].sum()
    df['market_share'] = df['total_category_revenue'] / total_revenue
    
    # Classifier les catégories par performance
    df['category_tier'] = pd.qcut(df['total_category_revenue'], 3, labels=['LOW', 'MEDIUM', 'HIGH'])
    
    df.to_sql('category_features', engine, if_exists='replace', index=False)
    
    logger.info("Table category_features créée avec %d catégories", len(df))


def create_elasticity_by_segment(engine: Engine) -> None:
    """
    Crée une table d'élasticité prix par segment/catégorie.
    Calcule une estimation de l'élasticité basée sur les données historiques.
    """
    logger.info("Création de la table elasticity_by_segment...")
    
    query = """
    WITH price_variation AS (
        SELECT 
            product_category_name,
            product_id,
            STDDEV(price) as price_std,
            AVG(price) as avg_price,
            COUNT(*) as n_orders
        FROM orders_enriched 
        WHERE price > 0 
        GROUP BY product_category_name, product_id
        HAVING COUNT(*) >= 5 AND STDDEV(price) > 0
    ),
    category_price_stats AS (
        SELECT 
            product_category_name,
            AVG(price_std) as avg_price_std,
            AVG(avg_price) as overall_avg_price,
            COUNT(DISTINCT product_id) as n_products,
            SUM(n_orders) as total_orders
        FROM price_variation
        GROUP BY product_category_name
    ),
    demand_variation AS (
        SELECT 
            product_category_name,
            EXTRACT(MONTH FROM order_purchase_timestamp) as month,
            COUNT(*) as monthly_orders,
            AVG(price) as monthly_avg_price
        FROM orders_enriched
        WHERE price > 0
        GROUP BY product_category_name, month
        HAVING COUNT(*) >= 10
    ),
    elasticity_estimation AS (
        SELECT 
            dv.product_category_name,
            -- Estimation basique de l'élasticité: corrélation prix/demande
            CORR(dv.monthly_orders, dv.monthly_avg_price) as price_demand_correlation,
            -- Élasticité estimée basée sur la variation des prix et de la demande
            CASE 
                WHEN STDDEV(dv.monthly_avg_price) > 0 THEN 
                    (STDDEV(dv.monthly_orders) / AVG(dv.monthly_orders)) / 
                    (STDDEV(dv.monthly_avg_price) / AVG(dv.monthly_avg_price))
                ELSE NULL 
            END as estimated_elasticity_raw
        FROM demand_variation dv
        GROUP BY dv.product_category_name
        HAVING COUNT(*) >= 3  -- Au moins 3 mois de données
    )
    SELECT 
        cps.product_category_name,
        cps.n_products,
        cps.total_orders,
        cps.overall_avg_price,
        cps.avg_price_std,
        ee.price_demand_correlation,
        COALESCE(
            -ABS(ee.estimated_elasticity_raw),  -- Élasticité négative par défaut
            CASE 
                WHEN cps.avg_price_std / cps.overall_avg_price > 0.3 THEN -1.5  -- Prix variables = élastique
                WHEN cps.avg_price_std / cps.overall_avg_price < 0.1 THEN -0.5  -- Prix stables = inélastique
                ELSE -1.0  -- Par défaut
            END
        ) as mean_elasticity,
        CASE 
            WHEN ABS(ee.price_demand_correlation) > 0.5 THEN 'HIGH_CONFIDENCE'
            WHEN ABS(ee.price_demand_correlation) > 0.2 THEN 'MEDIUM_CONFIDENCE' 
            ELSE 'LOW_CONFIDENCE'
        END as confidence_level
    FROM category_price_stats cps
    LEFT JOIN elasticity_estimation ee ON cps.product_category_name = ee.product_category_name
    WHERE cps.total_orders >= 50  -- Catégories avec suffisamment de données
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        # Nettoyage et validation des élasticités
        df['mean_elasticity'] = np.clip(df['mean_elasticity'], -3.0, -0.1)
        
        # Valeurs par défaut pour les catégories manquantes
        if len(df) == 0:
            logger.warning("Aucune donnée d'élasticité calculée, création de valeurs par défaut")
            default_query = """
            SELECT DISTINCT product_category_name 
            FROM orders_enriched 
            WHERE product_category_name IS NOT NULL
            """
            categories = pd.read_sql(default_query, engine)
            df = pd.DataFrame({
                'product_category_name': categories['product_category_name'],
                'mean_elasticity': -1.0,  # Élasticité moyenne par défaut
                'confidence_level': 'LOW_CONFIDENCE',
                'n_products': 1,
                'total_orders': 10
            })
        
        df.to_sql('elasticity_by_segment', engine, if_exists='replace', index=False)
        
        # Créer un index
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_elasticity_category ON elasticity_by_segment(product_category_name);
            """))
        
        logger.info("Table elasticity_by_segment créée avec %d segments", len(df))
        
        # Log des statistiques d'élasticité
        logger.info("Statistiques d'élasticité:")
        logger.info("  Moyenne: %.3f", df['mean_elasticity'].mean())
        logger.info("  Min: %.3f", df['mean_elasticity'].min())
        logger.info("  Max: %.3f", df['mean_elasticity'].max())
        logger.info("  Niveaux de confiance: %s", df['confidence_level'].value_counts().to_dict())
        
    except Exception as e:
        logger.error("Erreur lors de la création de elasticity_by_segment: %s", e)
        # Créer une table par défaut en cas d'erreur
        create_default_elasticity_table(engine)


def create_default_elasticity_table(engine: Engine) -> None:
    """Crée une table d'élasticité par défaut en cas d'erreur."""
    logger.info("Création d'une table d'élasticité par défaut...")
    
    query = """
    SELECT DISTINCT product_category_name 
    FROM orders_enriched 
    WHERE product_category_name IS NOT NULL
    """
    
    categories = pd.read_sql(query, engine)
    
    # Élasticités par défaut basées sur le type de catégorie
    def get_default_elasticity(category_name):
        category_lower = category_name.lower() if category_name else ""
        
        # Catégories probablement élastiques (biens non essentiels)
        elastic_categories = ['luxury', 'fashion', 'electronics', 'entertainment', 'sports']
        # Catégories probablement inélastiques (biens essentiels)
        inelastic_categories = ['food', 'health', 'household', 'baby', 'grocery']
        
        if any(elastic in category_lower for elastic in elastic_categories):
            return -1.8
        elif any(inelastic in category_lower for inelastic in inelastic_categories):
            return -0.6
        else:
            return -1.2  # Valeur par défaut
    
    df = pd.DataFrame({
        'product_category_name': categories['product_category_name'],
        'mean_elasticity': categories['product_category_name'].apply(get_default_elasticity),
        'confidence_level': 'ESTIMATED',
        'n_products': 10,
        'total_orders': 100
    })
    
    df.to_sql('elasticity_by_segment', engine, if_exists='replace', index=False)
    logger.info("Table elasticity_by_segment par défaut créée avec %d catégories", len(df))


def create_demand_features(engine: Engine) -> None:
    """
    Crée une table temporelle pour l'analyse de la demande.
    - Ventes par période (jour, semaine, mois)
    - Trends saisonnières
    - Élasticité temporelle
    """
    logger.info("Création de la table demand_features...")
    
    query = """
    WITH daily_sales AS (
        SELECT 
            DATE(order_purchase_timestamp) as sale_date,
            product_category_name,
            COUNT(*) as daily_orders,
            SUM(price) as daily_revenue,
            AVG(price) as avg_daily_price,
            COUNT(DISTINCT product_id) as unique_products_sold
        FROM orders_enriched
        WHERE order_purchase_timestamp IS NOT NULL
        GROUP BY DATE(order_purchase_timestamp), product_category_name
    ),
    weekly_aggregates AS (
        SELECT 
            EXTRACT(YEAR FROM sale_date) as year,
            EXTRACT(WEEK FROM sale_date) as week,
            product_category_name,
            AVG(daily_orders) as avg_weekly_orders,
            SUM(daily_revenue) as weekly_revenue,
            AVG(avg_daily_price) as avg_weekly_price
        FROM daily_sales
        GROUP BY year, week, product_category_name
    )
    SELECT 
        ds.*,
        wa.avg_weekly_orders,
        wa.weekly_revenue,
        wa.avg_weekly_price,
        EXTRACT(YEAR FROM ds.sale_date) as year,
        EXTRACT(MONTH FROM ds.sale_date) as month,
        EXTRACT(DOW FROM ds.sale_date) as day_of_week
    FROM daily_sales ds
    LEFT JOIN weekly_aggregates wa ON 
        EXTRACT(YEAR FROM ds.sale_date) = wa.year AND 
        EXTRACT(WEEK FROM ds.sale_date) = wa.week AND
        ds.product_category_name = wa.product_category_name
    WHERE ds.sale_date >= (SELECT MAX(sale_date) FROM daily_sales) - INTERVAL '365 days'
    """
    
    df = pd.read_sql(query, engine)
    
    # CORRECTION : Convertir sale_date en datetime
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    # Calculer des features de tendance
    df = df.sort_values(['product_category_name', 'sale_date'])
    
    # Features de lag pour time series
    df['orders_7d_lag'] = df.groupby('product_category_name')['daily_orders'].shift(7)
    df['price_7d_lag'] = df.groupby('product_category_name')['avg_daily_price'].shift(7)
    
    # Rolling averages
    df['orders_7d_avg'] = df.groupby('product_category_name')['daily_orders'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df['price_7d_avg'] = df.groupby('product_category_name')['avg_daily_price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Trends
    df['orders_trend'] = df['daily_orders'] / df['orders_7d_avg']
    df['price_trend'] = df['avg_daily_price'] / df['price_7d_avg']
    
    # CORRECTION : Features saisonnières avec conversion datetime
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_end'] = (df['sale_date'].dt.is_month_end).astype(int)
    
    df.to_sql('demand_features', engine, if_exists='replace', index=False)
    
    # Index pour les requêtes temporelles
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_demand_features_date ON demand_features(sale_date);
            CREATE INDEX IF NOT EXISTS idx_demand_features_category ON demand_features(product_category_name);
        """))
    
    logger.info("Table demand_features créée avec %d enregistrements", len(df))


def create_competition_features(engine: Engine) -> None:
    """
    Crée des features liées à la concurrence par produit.
    - Nombre de vendeurs par produit
    - Écart de prix entre vendeurs
    - Market share des vendeurs
    """
    logger.info("Création de la table competition_features...")
    
    query = """
    WITH seller_products AS (
        SELECT 
            product_id,
            COUNT(DISTINCT seller_id) as num_sellers,
            AVG(price) as market_avg_price,
            STDDEV(price) as market_price_std,
            MIN(price) as market_min_price,
            MAX(price) as market_max_price
        FROM orders_enriched
        GROUP BY product_id
    ),
    seller_market_share AS (
        SELECT 
            product_id,
            seller_id,
            COUNT(*) as seller_orders,
            SUM(price) as seller_revenue,
            AVG(price) as seller_avg_price
        FROM orders_enriched
        GROUP BY product_id, seller_id
    ),
    product_leader AS (
        SELECT 
            sms.product_id,
            sms.seller_id as leading_seller,
            sms.seller_avg_price as leader_price,
            sms.seller_orders as leader_orders,
            (sms.seller_orders * 1.0 / total.total_orders) as leader_market_share
        FROM seller_market_share sms
        INNER JOIN (
            SELECT product_id, MAX(seller_orders) as max_orders
            FROM seller_market_share
            GROUP BY product_id
        ) max_orders ON sms.product_id = max_orders.product_id AND sms.seller_orders = max_orders.max_orders
        INNER JOIN (
            SELECT product_id, SUM(seller_orders) as total_orders
            FROM seller_market_share
            GROUP BY product_id
        ) total ON sms.product_id = total.product_id
    )
    SELECT 
        sp.*,
        pl.leading_seller,
        pl.leader_price,
        pl.leader_orders,
        pl.leader_market_share,
        (sp.market_price_std / NULLIF(sp.market_avg_price, 0)) as price_variability_index,
        (sp.market_max_price - sp.market_min_price) as price_range_absolute,
        CASE 
            WHEN sp.num_sellers = 1 THEN 'MONOPOLY'
            WHEN sp.num_sellers BETWEEN 2 AND 3 THEN 'OLIGOPOLY'
            ELSE 'COMPETITIVE'
        END as market_structure
    FROM seller_products sp
    LEFT JOIN product_leader pl ON sp.product_id = pl.product_id
    WHERE sp.num_sellers > 0
    """
    
    df = pd.read_sql(query, engine)
    
    # Calculer la position compétitive
    df['competition_intensity'] = np.log1p(df['num_sellers'])
    df['price_discretion_power'] = 1 / (1 + df['price_variability_index'])
    
    df.to_sql('competition_features', engine, if_exists='replace', index=False)
    
    logger.info("Table competition_features créée avec %d produits", len(df))


def create_master_features_table(engine: Engine) -> None:
    """
    Crée une table finale unifiée avec toutes les features pour le modeling.
    """
    logger.info("Création de la table master_features...")
    
    query = """
    SELECT 
        oe.*,
        pf.total_orders as product_total_orders,
        pf.avg_price as product_avg_price,
        pf.median_price as product_median_price,
        pf.price_std as product_price_std,
        pf.sales_last_30d,
        pf.sales_last_90d,
        pf.product_status,
        pf.price_to_freight_ratio,
        cf.avg_category_price,
        cf.market_share as category_market_share,
        cf.category_tier,
        comp.num_sellers,
        comp.market_avg_price,
        comp.market_price_std,
        comp.market_structure,
        comp.competition_intensity,
        es.mean_elasticity,
        es.confidence_level as elasticity_confidence,
        -- Features temporelles
        EXTRACT(YEAR FROM oe.order_purchase_timestamp) as order_year,
        EXTRACT(MONTH FROM oe.order_purchase_timestamp) as order_month,
        EXTRACT(DOW FROM oe.order_purchase_timestamp) as order_day_of_week,
        -- Target variable pour le modèle de demande
        CASE WHEN oe.price > 0 THEN 1 ELSE 0 END as is_purchased
    FROM orders_enriched oe
    LEFT JOIN product_features pf ON oe.product_id = pf.product_id
    LEFT JOIN category_features cf ON oe.product_category_name = cf.product_category_name
    LEFT JOIN competition_features comp ON oe.product_id = comp.product_id
    LEFT JOIN elasticity_by_segment es ON oe.product_category_name = es.product_category_name
    WHERE oe.price IS NOT NULL AND oe.price > 0
    """
    
    df = pd.read_sql(query, engine)
    
    # CORRECTION : Convertir les colonnes datetime
    datetime_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_customer_date']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Nettoyage final
    df = df.fillna({
        'product_total_orders': 0,
        'product_avg_price': df['price'].mean(),
        'product_price_std': 0,
        'sales_last_30d': 0,
        'avg_category_price': df['price'].mean(),
        'num_sellers': 1,
        'market_avg_price': df['price'].mean(),
        'mean_elasticity': -1.0  # Élasticité par défaut
    })
    
    # Features dérivées finales
    df['price_ratio_to_avg'] = df['price'] / df['product_avg_price']
    df['price_ratio_to_category'] = df['price'] / df['avg_category_price']
    df['price_ratio_to_market'] = df['price'] / df['market_avg_price']
    df['is_weekend'] = (df['order_day_of_week'] >= 5).astype(int)
    
    # CORRECTION : Feature mois fin avec conversion datetime
    if 'order_purchase_timestamp' in df.columns:
        df['is_month_end'] = (df['order_purchase_timestamp'].dt.is_month_end).astype(int)
    
    # Sauvegarde la table finale
    df.to_sql('master_features', engine, if_exists='replace', index=False)
    
    # Créer les index pour le modeling
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_master_features_product ON master_features(product_id);
            CREATE INDEX IF NOT EXISTS idx_master_features_category ON master_features(product_category_name);
            CREATE INDEX IF NOT EXISTS idx_master_features_date ON master_features(order_purchase_timestamp);
            CREATE INDEX IF NOT EXISTS idx_master_features_price ON master_features(price);
        """))
    
    logger.info("Table master_features créée avec %d enregistrements", len(df))


def run_all_transformations(engine: Engine) -> None:
    """Exécute toutes les transformations dans l'ordre."""
    logger.info("Début des transformations...")
    
    transformations = [
        create_product_features,
        create_category_features,
        create_elasticity_by_segment,  # NOUVELLE TABLE AJOUTÉE
        create_demand_features,
        create_competition_features,
        create_master_features_table
    ]
    
    for transformation in transformations:
        try:
            transformation(engine)
            logger.info("✓ %s terminé", transformation.__name__)
        except Exception as e:
            logger.error("✗ Erreur dans %s: %s", transformation.__name__, e)
            # Continuer avec les transformations suivantes
            continue
    
    logger.info("Toutes les transformations terminées!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform Olist data for pricing analysis")
    parser.add_argument("--pg-uri", type=str, default=os.environ.get("POSTGRES_URI"),
                       help="URI Postgres (ex: postgresql://user:pass@host:5432/dbname)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.pg_uri:
        logger.error("Aucune PG URI fournie. Passez --pg-uri ou définissez POSTGRES_URI")
        raise SystemExit(1)
    
    engine = get_engine(args.pg_uri)
    run_all_transformations(engine)


if __name__ == "__main__":
    main()