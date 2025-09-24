import sys
import os
sys.path.append(os.path.dirname(__file__))

from elasticity import PriceOptimizer, ElasticitySimulator
import pandas as pd
import matplotlib.pyplot as plt

def test_price_optimizer():
    """Teste l'optimiseur de prix."""
    print("=== TEST OPTIMISEUR DE PRIX ===")
    
    optimizer = PriceOptimizer(demand_model=None)
    
    # Cas 1: Produit élastique (elasticité = -2.0)
    result = optimizer.find_optimal_price(
        base_price=100, 
        base_demand=1000, 
        elasticity=-2.0,
        cost=50
    )
    
    print("Cas élastique (elasticité = -2.0):")
    print(f"Prix actuel: 100€, Demande: 1000 units")
    print(f"Prix optimal: {result['optimal_price']:.2f}€")
    print(f"Gain revenue: {result['revenue_gain_pct']:.1f}%")
    print()

def test_elasticity_simulator():
    """Teste le simulateur d'élasticité."""
    print("=== TEST SIMULATEUR ÉLASTICITÉ ===")
    
    simulator = ElasticitySimulator()
    
    # Simulation pour un produit
    results = simulator.simulate_price_change(
        base_price=50,
        base_demand=500,
        elasticity=-1.5
    )
    
    print("Impact des changements de prix (elasticité = -1.5):")
    print(results.head(10))
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(results['price_change_pct'], results['revenue_change_pct'], marker='o')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Changement de Prix (%)')
    plt.ylabel('Changement de Revenue (%)')
    plt.title('Impact des Changements de Prix sur le Revenue')
    plt.grid(True, alpha=0.3)
    plt.savefig('../../outputs/elasticity_simulation.png')
    plt.show()

def test_profit_maximization():
    """Teste la maximisation du profit."""
    print("=== TEST MAXIMISATION PROFIT ===")
    
    simulator = ElasticitySimulator()
    
    result = simulator.find_profit_maximizing_price(
        base_price=80,
        base_demand=300,
        elasticity=-1.2,
        unit_cost=40
    )
    
    print("Maximisation du profit:")
    print(f"Prix actuel: 80€, Coût unitaire: 40€")
    print(f"Prix optimal: {result['optimal_price']:.2f}€")
    print(f"Gain profit: {result['price_change_pct']:.1f}%")

if __name__ == "__main__":
    test_price_optimizer()
    test_elasticity_simulator()
    test_profit_maximization()