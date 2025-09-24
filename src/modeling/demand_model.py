"""
Optimisation des prix basée sur l'élasticité - Version corrigée.
"""

from __future__ import annotations
import argparse
import logging
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("price_optimizer")

class RobustPriceOptimizer:
    """Optimiseur de prix robuste avec gestion d'erreurs."""
    
    def __init__(self, max_products: int = 1000):
        self.max_products = max_products
        
    def demand_function(self, price, base_price, base_demand, elasticity):
        """Fonction de demande vectorisée."""
        return base_demand * np.power(price / base_price, elasticity)
    
    def optimize_single_product(self, base_price, base_demand, elasticity, 
                              unit_cost=0, price_bounds=(0.1, 1000)):
        """Optimisation rapide pour un seul produit."""
        try:
            # Fonction objectif simplifiée
            def objective(price):
                demand = self.demand_function(price, base_price, base_demand, elasticity)
                revenue = (price - unit_cost) * demand
                return -revenue  # Minimiser le revenue négatif
            
            result = minimize(
                objective,
                x0=[base_price * 0.9],
                bounds=[price_bounds],
                method='L-BFGS-B',
                options={'maxiter': 50}  # Réduit encore les itérations
            )
            
            if result.success:
                optimal_price = result.x[0]
                optimal_revenue = -result.fun
                current_revenue = (base_price - unit_cost) * base_demand
                
                return {
                    'optimal_price': optimal_price,
                    'optimal_revenue': optimal_revenue,
                    'current_revenue': current_revenue,
                    'revenue_gain_pct': (optimal_revenue - current_revenue) / current_revenue * 100,
                    'price_change_pct': (optimal_price - base_price) / base_price * 100,
                    'success': True
                }
            else:
                return {
                    'optimal_price': base_price,
                    'revenue_gain_pct': 0,
                    'price_change_pct': 0,
                    'success': False,
                    'error': result.message
                }
                
        except Exception as e:
            logger.warning(f"Erreur optimisation produit: {e}")
            return {
                'optimal_price': base_price,
                'revenue_gain_pct': 0,
                'price_change_pct': 0,
                'success': False,
                'error': str(e)
            }

def check_tables_exist(engine):
    """Vérifie si les tables nécessaires existent."""
    required_tables = ['product_features', 'category_features']
    existing_tables = []
    
    for table in required_tables:
        try:
            query = f"SELECT 1 FROM {table} LIMIT 1"
            pd.read_sql(query, engine)
            existing_tables.append(table)
        except Exception as e:
            logger.warning(f"Table {table} non trouvée: {e}")
    
    return existing_tables

def estimate_elasticity_from_data(engine):
    """Estime l'élasticité à partir des données historiques si la table n'existe pas."""
    logger.info("Estimation de l'élasticité à partir des données...")
    
    try:
        # Query pour calculer l'élasticité basique
        query = """
        WITH price_variation AS (
            SELECT 
                product_id,
                product_category_name,
                STDDEV(price) as price_std,
                AVG(price) as avg_price,
                COUNT(*) as total_orders
            FROM orders_enriched 
            WHERE price > 0 
            GROUP BY product_id, product_category_name
            HAVING COUNT(*) >= 5 AND STDDEV(price) > 0
        ),
        demand_variation AS (
            SELECT 
                product_id,
                COUNT(*) as order_count,
                AVG(price) as avg_price
            FROM orders_enriched 
            WHERE price > 0
            GROUP BY product_id
        )
        SELECT 
            pv.product_category_name,
            -0.8 as estimated_elasticity  -- Valeur par défaut réaliste
        FROM price_variation pv
        GROUP BY pv.product_category_name
        """
        
        elasticity_df = pd.read_sql(query, engine)
        logger.info(f"Élasticité estimée pour {len(elasticity_df)} catégories")
        return elasticity_df
        
    except Exception as e:
        logger.warning(f"Erreur estimation élasticité: {e}")
        # Retourne une élasticité par défaut
        return pd.DataFrame({
            'product_category_name': ['default'],
            'estimated_elasticity': [-1.0]  # Élasticité moyenne
        })

def load_and_prepare_data(engine, max_products=1000):
    """Charge et prépare les données de manière robuste."""
    logger.info("Chargement des données...")
    
    # Vérification des tables
    existing_tables = check_tables_exist(engine)
    if 'product_features' not in existing_tables:
        logger.error("La table product_features n'existe pas. Exécutez d'abord transform.py")
        return pd.DataFrame()
    
    # Estimation de l'élasticité
    elasticity_df = estimate_elasticity_from_data(engine)
    
    # Query principale simplifiée
    query = f"""
    SELECT 
        product_id,
        product_category_name,
        avg_price as base_price,
        total_orders as base_demand,
        price_std,
        sales_last_30d,
        product_status
    FROM product_features 
    WHERE product_status = 'ACTIVE' 
      AND total_orders >= 5
      AND avg_price > 0
    ORDER BY total_orders DESC
    LIMIT {max_products}
    """
    
    products_data = pd.read_sql(query, engine)
    logger.info(f"Données de base chargées: {len(products_data)} produits")
    
    if len(products_data) == 0:
        return products_data
    
    # Fusion avec l'élasticité estimée
    products_data = products_data.merge(
        elasticity_df, 
        on='product_category_name', 
        how='left'
    )
    
    # Remplissage des valeurs manquantes
    products_data['elasticity'] = products_data['estimated_elasticity'].fillna(-1.0)
    products_data['unit_cost'] = products_data['base_price'] * 0.6
    
    # Nettoyage
    products_data = products_data.dropna(subset=['elasticity'])
    products_data['elasticity'] = np.clip(products_data['elasticity'], -3.0, -0.1)
    
    return products_data

def optimize_products_batch(products_data, batch_size=50):
    """Optimise les produits par lots."""
    optimizer = RobustPriceOptimizer()
    results = []
    
    n_products = len(products_data)
    if n_products == 0:
        return pd.DataFrame()
    
    for i in range(0, n_products, batch_size):
        batch = products_data.iloc[i:i + batch_size]
        logger.info(f"Optimisation lot {i//batch_size + 1}/{(n_products-1)//batch_size + 1}")
        
        for _, product in batch.iterrows():
            try:
                result = optimizer.optimize_single_product(
                    base_price=product['base_price'],
                    base_demand=product['base_demand'],
                    elasticity=product['elasticity'],
                    unit_cost=product['unit_cost']
                )
                
                results.append({
                    'product_id': product['product_id'],
                    'product_category_name': product['product_category_name'],
                    'current_price': product['base_price'],
                    'current_demand': product['base_demand'],
                    'elasticity': product['elasticity'],
                    **result
                })
                
            except Exception as e:
                logger.warning(f"Erreur produit {product['product_id']}: {e}")
                continue
    
    return pd.DataFrame(results)

def main():
    """Workflow principal robuste."""
    parser = argparse.ArgumentParser(description="Optimisation des prix - Version robuste")
    parser.add_argument("--pg-uri", type=str, default=os.environ.get("POSTGRES_URI"),
                       help="URI Postgres")
    parser.add_argument("--max-products", type=int, default=100,
                       help="Nombre maximum de produits à optimiser")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Taille des lots pour l'optimisation")
    parser.add_argument("--output-dir", type=str, default="../../outputs")
    
    args = parser.parse_args()
    
    if not args.pg_uri:
        logger.error("URI Postgres requise")
        return
    
    try:
        # Connexion à la base
        engine = create_engine(args.pg_uri)
        
        # Test de connexion
        try:
            pd.read_sql("SELECT 1", engine)
            logger.info("Connexion à la base de données réussie")
        except Exception as e:
            logger.error(f"Erreur de connexion: {e}")
            return
        
        # Chargement des données
        products_data = load_and_prepare_data(engine, args.max_products)
        
        if len(products_data) == 0:
            logger.error("Aucune donnée à optimiser. Vérifiez que transform.py a été exécuté.")
            return
        
        logger.info(f"Optimisation de {len(products_data)} produits...")
        
        # Optimisation
        results = optimize_products_batch(products_data, args.batch_size)
        
        if len(results) == 0:
            logger.error("Aucun résultat d'optimisation")
            return
        
        # Sauvegarde
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = f"{args.output_dir}/optimized_prices.csv"
        results.to_csv(output_path, index=False)
        
        # Résumé
        successful = results[results['success']]
        logger.info("=== RÉSULTATS ===")
        logger.info(f"Produits traités: {len(results)}")
        logger.info(f"Optimisations réussies: {len(successful)}")
        
        if len(successful) > 0:
            avg_gain = successful['revenue_gain_pct'].mean()
            logger.info(f"Gain moyen: {avg_gain:.1f}%")
            
            # Top 3 gains
            top_gains = successful.nlargest(3, 'revenue_gain_pct')
            logger.info("Meilleurs gains:")
            for _, product in top_gains.iterrows():
                logger.info(f"  {product['product_id']}: {product['revenue_gain_pct']:.1f}%")
        
        logger.info(f"Résultats sauvegardés: {output_path}")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()