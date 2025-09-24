"""
Simulation de tests A/B pour valider les stratégies de prix.
Usage:
    python simulate_ab.py --pg-uri postgresql://user:pass@localhost:5432/dbname
"""

from __future__ import annotations
import argparse
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ab_test_simulation")


class ABTestSimulator:
    """Simulateur de tests A/B pour l'optimisation des prix."""
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        self.alpha = alpha  # Niveau de significativité
        self.power = power  # Puissance statistique
        self.results = {}
        
    def calculate_sample_size(self, baseline_conversion: float, mde: float) -> int:
        """
        Calcule la taille d'échantillon nécessaire pour détecter un effet.
        
        Args:
            baseline_conversion: Taux de conversion de base
            mde: Minimum Detectable Effect (effet minimum détectable)
            
        Returns:
            Taille d'échantillon par groupe
        """
        # Z-scores pour alpha et power
        z_alpha = stats.norm.ppf(1 - self.alpha/2)  # Test bilatéral
        z_beta = stats.norm.ppf(self.power)
        
        # Calcul de la taille d'échantillon
        p1 = baseline_conversion
        p2 = baseline_conversion * (1 + mde)
        p_pool = (p1 + p2) / 2
        
        numerator = (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + 
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p1 - p2) ** 2
        
        return int(np.ceil(numerator / denominator))
    
    def simulate_conversion_ab_test(self, baseline_rate: float, treatment_effect: float,
                                  sample_size: int, n_simulations: int = 1000) -> Dict:
        """
        Simule un test A/B de conversion.
        
        Returns:
            Probabilité de détection, puissance, etc.
        """
        significant_detections = 0
        p_values = []
        effect_sizes = []
        
        for _ in range(n_simulations):
            # Génération des données
            control_conversions = np.random.binomial(sample_size, baseline_rate)
            treatment_conversions = np.random.binomial(sample_size, baseline_rate * (1 + treatment_effect))
            
            # Test statistique
            control_rate = control_conversions / sample_size
            treatment_rate = treatment_conversions / sample_size
            
            # Test de proportion
            z_stat, p_value = self.proportion_z_test(control_conversions, sample_size,
                                                    treatment_conversions, sample_size)
            
            p_values.append(p_value)
            effect_sizes.append(treatment_rate - control_rate)
            
            if p_value < self.alpha:
                significant_detections += 1
        
        power = significant_detections / n_simulations
        
        return {
            'power': power,
            'avg_p_value': np.mean(p_values),
            'avg_effect_size': np.mean(effect_sizes),
            'detection_probability': power,
            'required_sample_size': self.calculate_sample_size(baseline_rate, treatment_effect)
        }
    
    def proportion_z_test(self, successes_a: int, n_a: int, 
                         successes_b: int, n_b: int) -> Tuple[float, float]:
        """Test Z pour deux proportions."""
        p_a = successes_a / n_a
        p_b = successes_b / n_b
        p_pool = (successes_a + successes_b) / (n_a + n_b)
        
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z = (p_a - p_b) / se
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p_value
    
    def simulate_revenue_ab_test(self, baseline_data: pd.DataFrame,
                               price_changes: Dict[str, float],
                               elasticity: float, n_simulations: int = 1000) -> pd.DataFrame:
        """
        Simule un test A/B de revenue avec changements de prix.
        
        Args:
            baseline_data: Données historiques des produits
            price_changes: Dictionnaire product_id -> changement de prix (%)
            elasticity: Élasticité prix moyenne
        """
        results = []
        
        for product_id, price_change_pct in price_changes.items():
            product_data = baseline_data[baseline_data['product_id'] == product_id]
            
            if len(product_data) == 0:
                continue
                
            baseline_price = product_data['avg_price'].iloc[0]
            baseline_demand = product_data['total_orders'].iloc[0]
            baseline_revenue = baseline_price * baseline_demand
            
            # Simulation des résultats
            simulated_revenues_control = []
            simulated_revenues_treatment = []
            
            for _ in range(n_simulations):
                # Groupe contrôle (prix actuel)
                demand_control = np.random.poisson(baseline_demand)
                revenue_control = baseline_price * demand_control
                simulated_revenues_control.append(revenue_control)
                
                # Groupe traitement (nouveau prix)
                new_price = baseline_price * (1 + price_change_pct / 100)
                demand_change = elasticity * (price_change_pct / 100)
                new_demand = baseline_demand * (1 + demand_change)
                demand_treatment = max(0, np.random.poisson(new_demand))
                revenue_treatment = new_price * demand_treatment
                simulated_revenues_treatment.append(revenue_treatment)
            
            # Analyse statistique
            mean_control = np.mean(simulated_revenues_control)
            mean_treatment = np.mean(simulated_revenues_treatment)
            revenue_change_pct = (mean_treatment - mean_control) / mean_control * 100
            
            # Test t
            t_stat, p_value = stats.ttest_ind(simulated_revenues_treatment, 
                                            simulated_revenues_control)
            
            # Calcul de la puissance
            effect_size = (mean_treatment - mean_control) / np.std(simulated_revenues_control)
            required_sample = self.calculate_revenue_sample_size(
                np.std(simulated_revenues_control), 
                mean_treatment - mean_control
            )
            
            results.append({
                'product_id': product_id,
                'price_change_pct': price_change_pct,
                'baseline_revenue': baseline_revenue,
                'expected_revenue_change_pct': revenue_change_pct,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'effect_size': effect_size,
                'required_sample_size': required_sample,
                'mean_control': mean_control,
                'mean_treatment': mean_treatment
            })
        
        return pd.DataFrame(results)
    
    def calculate_revenue_sample_size(self, std_dev: float, mde: float) -> int:
        """Calcule la taille d'échantillon pour un test de revenue."""
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(self.power)
        
        n = ((z_alpha + z_beta) ** 2 * (2 * std_dev ** 2)) / (mde ** 2)
        return int(np.ceil(n))
    
    def run_product_ab_simulation(self, products_data: pd.DataFrame,
                                optimized_prices: pd.DataFrame,
                                simulation_days: int = 30) -> pd.DataFrame:
        """
        Simule un test A/B complet pour un portefeuille de produits.
        """
        ab_results = []
        
        for _, product in optimized_prices.iterrows():
            product_id = product['product_id']
            current_price = product['current_price']
            optimized_price = product['optimal_price']
            elasticity = product['elasticity']
            
            # Données historiques du produit
            hist_data = products_data[products_data['product_id'] == product_id]
            if len(hist_data) == 0:
                continue
                
            baseline_daily_demand = hist_data['total_orders'].iloc[0] / 365  # Demande journalière estimée
            
            # Simulation des groupes A/B
            control_revenues = []
            treatment_revenues = []
            control_conversions = []
            treatment_conversions = []
            
            for day in range(simulation_days):
                # Groupe contrôle (prix actuel)
                daily_demand_control = np.random.poisson(baseline_daily_demand)
                control_revenues.append(current_price * daily_demand_control)
                control_conversions.append(daily_demand_control)
                
                # Groupe traitement (prix optimisé)
                demand_change = elasticity * ((optimized_price - current_price) / current_price)
                daily_demand_treatment = np.random.poisson(
                    max(0, baseline_daily_demand * (1 + demand_change))
                )
                treatment_revenues.append(optimized_price * daily_demand_treatment)
                treatment_conversions.append(daily_demand_treatment)
            
            # Analyse des résultats
            avg_control_revenue = np.mean(control_revenues)
            avg_treatment_revenue = np.mean(treatment_revenues)
            revenue_lift = (avg_treatment_revenue - avg_control_revenue) / avg_control_revenue * 100
            
            # Tests statistiques
            t_stat_revenue, p_value_revenue = stats.ttest_ind(treatment_revenues, control_revenues)
            _, p_value_conversion = stats.ttest_ind(treatment_conversions, control_conversions)
            
            # Calcul de l'intervalle de confiance
            ci_low, ci_high = self.calculate_confidence_interval(
                treatment_revenues, control_revenues
            )
            
            ab_results.append({
                'product_id': product_id,
                'current_price': current_price,
                'optimized_price': optimized_price,
                'price_change_pct': ((optimized_price - current_price) / current_price) * 100,
                'elasticity': elasticity,
                'revenue_lift_pct': revenue_lift,
                'p_value_revenue': p_value_revenue,
                'p_value_conversion': p_value_conversion,
                'significant_revenue': p_value_revenue < self.alpha,
                'significant_conversion': p_value_conversion < self.alpha,
                'confidence_interval_low': ci_low,
                'confidence_interval_high': ci_high,
                'expected_annual_gain': revenue_lift * avg_control_revenue * 365 / 100,
                'simulation_days': simulation_days,
                'avg_daily_demand': baseline_daily_demand
            })
        
        return pd.DataFrame(ab_results)
    
    def calculate_confidence_interval(self, treatment_data: List, control_data: List, 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calcule l'intervalle de confiance pour la différence des moyennes."""
        mean_treatment = np.mean(treatment_data)
        mean_control = np.mean(control_data)
        std_treatment = np.std(treatment_data, ddof=1)
        std_control = np.std(control_data, ddof=1)
        n_treatment = len(treatment_data)
        n_control = len(control_data)
        
        # Erreur standard de la différence
        se_diff = np.sqrt((std_treatment**2 / n_treatment) + (std_control**2 / n_control))
        
        # Score Z pour l'intervalle de confiance
        z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        margin_of_error = z_score * se_diff
        difference = mean_treatment - mean_control
        
        return (difference - margin_of_error, difference + margin_of_error)
    
    def generate_ab_test_recommendations(self, ab_results: pd.DataFrame) -> pd.DataFrame:
        """Génère des recommandations basées sur les résultats A/B."""
        recommendations = []
        
        for _, result in ab_results.iterrows():
            if result['significant_revenue']:
                if result['revenue_lift_pct'] > 5:
                    recommendation = "IMPLEMENT"
                    confidence = "HIGH"
                    reason = "Gain de revenue significatif et important"
                elif result['revenue_lift_pct'] > 0:
                    recommendation = "CONSIDER"
                    confidence = "MEDIUM"
                    reason = "Gain de revenue modéré mais significatif"
                else:
                    recommendation = "REJECT"
                    confidence = "HIGH"
                    reason = "Perte de revenue significative"
            else:
                if result['revenue_lift_pct'] > 10:
                    recommendation = "TEST_LONGER"
                    confidence = "LOW"
                    reason = "Gain potentiel important mais non significatif - prolonger le test"
                else:
                    recommendation = "HOLD"
                    confidence = "LOW"
                    reason = "Impact non significatif - besoin de plus de données"
            
            recommendations.append({
                'product_id': result['product_id'],
                'recommendation': recommendation,
                'confidence': confidence,
                'reason': reason,
                'expected_annual_gain': result['expected_annual_gain'],
                'revenue_lift_pct': result['revenue_lift_pct'],
                'p_value': result['p_value_revenue']
            })
        
        return pd.DataFrame(recommendations)


class ABTestVisualizer:
    """Visualise les résultats des tests A/B."""
    
    def __init__(self, output_dir: str = "../outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_ab_test_dashboard(self, ab_results: pd.DataFrame, 
                               recommendations: pd.DataFrame) -> None:
        """Crée un dashboard visuel des résultats A/B."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution des gains de revenue
        axes[0, 0].hist(ab_results['revenue_lift_pct'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Gain de Revenue (%)')
        axes[0, 0].set_ylabel('Nombre de Produits')
        axes[0, 0].set_title('Distribution des Gains de Revenue')
        
        # 2. Significance vs Effect Size
        significant = ab_results['significant_revenue']
        axes[0, 1].scatter(ab_results['revenue_lift_pct'], -np.log10(ab_results['p_value_revenue']),
                          c=significant, cmap='viridis', alpha=0.6)
        axes[0, 1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[0, 1].set_xlabel('Gain de Revenue (%)')
        axes[0, 1].set_ylabel('-log10(p-value)')
        axes[0, 1].set_title('Significativité vs Effet')
        axes[0, 1].legend()
        
        # 3. Recommendations par catégorie
        rec_counts = recommendations['recommendation'].value_counts()
        axes[1, 0].pie(rec_counts.values, labels=rec_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Répartition des Recommandations')
        
        # 4. Gain annuel attendu par recommandation
        gain_by_rec = recommendations.groupby('recommendation')['expected_annual_gain'].sum()
        axes[1, 1].bar(gain_by_rec.index, gain_by_rec.values)
        axes[1, 1].set_title('Gain Annuel par Type de Recommandation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ab_test_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_sensitivity_analysis(self, simulator: ABTestSimulator,
                                  baseline_conversion: float,
                                  effect_sizes: List[float]) -> None:
        """Analyse de sensibilité pour la taille d'échantillon."""
        sample_sizes = []
        
        for effect in effect_sizes:
            n = simulator.calculate_sample_size(baseline_conversion, effect)
            sample_sizes.append(n)
        
        plt.figure(figsize=(10, 6))
        plt.plot(effect_sizes, sample_sizes, marker='o')
        plt.xlabel('Effet Détectable (%)')
        plt.ylabel('Taille d\'Échantillon Requise')
        plt.title('Sensibilité: Effet Détectable vs Taille d\'Échantillon')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Workflow principal de simulation A/B."""
    parser = argparse.ArgumentParser(description="Simulation de tests A/B pour la validation des prix")
    parser.add_argument("--pg-uri", type=str, default=os.environ.get("POSTGRES_URI"),
                       help="URI Postgres")
    parser.add_argument("--optimized-prices-path", type=str, 
                       default="../../outputs/optimized_prices.csv",
                       help="Chemin vers les prix optimisés")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Niveau de significativité")
    parser.add_argument("--power", type=float, default=0.8,
                       help="Puissance statistique")
    parser.add_argument("--simulation-days", type=int, default=30,
                       help="Durée de simulation en jours")
    parser.add_argument("--output-dir", type=str, default="../../outputs",
                       help="Répertoire de sortie")
    args = parser.parse_args()
    
    if not args.pg_uri:
        logger.error("URI Postgres requise")
        raise SystemExit(1)
    
    try:
        # Initialisation
        simulator = ABTestSimulator(alpha=args.alpha, power=args.power)
        visualizer = ABTestVisualizer(args.output_dir)
        
        # Chargement des données
        engine = create_engine(args.pg_uri)
        optimized_prices = pd.read_csv(args.optimized_prices_path)
        
        # Données produits
        query = """
        SELECT product_id, product_category_name, avg_price, total_orders, 
               price_std, sales_last_30d, product_status
        FROM product_features 
        WHERE product_status = 'ACTIVE'
        """
        products_data = pd.read_sql(query, engine)
        
        # Simulation A/B
        logger.info("Lancement de la simulation A/B...")
        ab_results = simulator.run_product_ab_simulation(
            products_data, optimized_prices, args.simulation_days
        )
        
        # Recommandations
        recommendations = simulator.generate_ab_test_recommendations(ab_results)
        
        # Analyse de sensibilité
        effect_sizes = [0.01, 0.05, 0.1, 0.15, 0.2]
        visualizer.create_sensitivity_analysis(simulator, 0.1, effect_sizes)
        
        # Sauvegarde des résultats
        ab_results.to_csv(f"{args.output_dir}/ab_test_results.csv", index=False)
        recommendations.to_csv(f"{args.output_dir}/ab_test_recommendations.csv", index=False)
        
        # Dashboard visuel
        visualizer.create_ab_test_dashboard(ab_results, recommendations)
        
        # Résumé
        logger.info("=== RÉSULTATS DE LA SIMULATION A/B ===")
        logger.info(f"Produits testés: {len(ab_results)}")
        logger.info(f"Gain de revenue moyen: {ab_results['revenue_lift_pct'].mean():.1f}%")
        
        significant_results = ab_results[ab_results['significant_revenue']]
        logger.info(f"Résultats significatifs: {len(significant_results)}")
        
        if len(significant_results) > 0:
            avg_significant_lift = significant_results['revenue_lift_pct'].mean()
            logger.info(f"Gain moyen significatif: {avg_significant_lift:.1f}%")
        
        # Résumé des recommandations
        rec_summary = recommendations['recommendation'].value_counts()
        logger.info("\n=== RECOMMANDATIONS ===")
        for rec, count in rec_summary.items():
            total_gain = recommendations[recommendations['recommendation'] == rec]['expected_annual_gain'].sum()
            logger.info(f"{rec}: {count} produits (gain annuel: {total_gain:,.0f}€)")
        
        # Produits prioritaires
        top_products = recommendations.nlargest(5, 'expected_annual_gain')
        logger.info("\n=== TOP 5 PRODUITS PRIORITAIRES ===")
        for _, product in top_products.iterrows():
            logger.info(f"{product['product_id']}: {product['expected_annual_gain']:,.0f}€ de gain annuel")
        
    except Exception as e:
        logger.error(f"Erreur lors de la simulation A/B: {e}")
        raise


if __name__ == "__main__":
    main()