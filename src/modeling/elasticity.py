import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger("elasticity_analysis")


class PriceOptimizer:
    """Optimiseur de prix basé sur l'élasticité."""
    
    def __init__(self, demand_model, cost_data=None):
        self.demand_model = demand_model
        self.cost_data = cost_data or {}
        
    def revenue_function(self, price, base_demand, elasticity, cost=0):
        """Fonction de revenue: price * demand(price)"""
        demand = base_demand * (price / self.base_price) ** elasticity
        revenue = (price - cost) * demand
        return -revenue  # Négatif pour la minimisation
    
    def find_optimal_price(self, base_price, base_demand, elasticity, cost=0, 
                         price_bounds=(0.1, 1000)):
        """Trouve le prix optimal pour maximiser le revenue."""
        self.base_price = base_price
        
        result = minimize(
            self.revenue_function,
            x0=base_price,
            args=(base_demand, elasticity, cost),
            bounds=[price_bounds],
            method='L-BFGS-B'
        )
        
        if result.success:
            optimal_price = result.x[0]
            optimal_revenue = -result.fun
            current_revenue = (base_price - cost) * base_demand
            
            return {
                'optimal_price': optimal_price,
                'optimal_revenue': optimal_revenue,
                'current_revenue': current_revenue,
                'revenue_gain_pct': (optimal_revenue - current_revenue) / current_revenue * 100,
                'price_change_pct': (optimal_price - base_price) / base_price * 100
            }
        else:
            logger.warning(f"Optimisation échouée: {result.message}")
            return None


class ElasticitySimulator:
    """Simulateur pour l'analyse de scénarios d'élasticité."""
    
    def __init__(self):
        self.scenarios = []
        
    def simulate_price_change(self, base_price, base_demand, elasticity, 
                            price_changes_pct=np.arange(-20, 21, 5)):
        """Simule l'impact de changements de prix sur le revenue."""
        results = []
        
        for change_pct in price_changes_pct:
            new_price = base_price * (1 + change_pct / 100)
            demand_change = elasticity * (change_pct / 100)
            new_demand = base_demand * (1 + demand_change)
            new_revenue = new_price * new_demand
            base_revenue = base_price * base_demand
            
            results.append({
                'price_change_pct': change_pct,
                'new_price': new_price,
                'demand_change_pct': demand_change * 100,
                'new_demand': new_demand,
                'revenue_change_pct': (new_revenue - base_revenue) / base_revenue * 100,
                'new_revenue': new_revenue
            })
        
        return pd.DataFrame(results)
    
    def find_profit_maximizing_price(self, base_price, base_demand, elasticity, unit_cost):
        """Trouve le prix maximisant le profit (revenu - coûts)."""
        prices = np.linspace(base_price * 0.5, base_price * 2, 100)
        profits = []
        
        for price in prices:
            demand = base_demand * (price / base_price) ** elasticity
            revenue = price * demand
            cost = unit_cost * demand
            profit = revenue - cost
            profits.append(profit)
        
        optimal_idx = np.argmax(profits)
        return {
            'optimal_price': prices[optimal_idx],
            'max_profit': profits[optimal_idx],
            'base_profit': (base_price - unit_cost) * base_demand,
            'price_change_pct': (prices[optimal_idx] - base_price) / base_price * 100
        }


def calculate_historical_elasticity(df, price_col='price', quantity_col='quantity', 
                                  time_col='date', product_col='product_id'):
    """Calcule l'élasticité historique à partir des données de ventes."""
    elasticity_results = []
    
    for product in df[product_col].unique():
        product_data = df[df[product_col] == product].sort_values(time_col)
        
        if len(product_data) > 1:
            price_changes = product_data[price_col].pct_change().dropna()
            quantity_changes = product_data[quantity_col].pct_change().dropna()
            
            if len(price_changes) > 0 and len(quantity_changes) > 0:
                # Élasticité = %Δquantity / %Δprice
                elasticity = (quantity_changes / price_changes).mean()
                
                elasticity_results.append({
                    'product_id': product,
                    'historical_elasticity': elasticity,
                    'n_observations': len(price_changes),
                    'avg_price': product_data[price_col].mean(),
                    'avg_quantity': product_data[quantity_col].mean()
                })
    
    return pd.DataFrame(elasticity_results)