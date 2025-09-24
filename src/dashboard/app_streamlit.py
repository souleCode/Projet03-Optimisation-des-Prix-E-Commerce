import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Optimisation Prix E-commerce",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .strategy-PENETRATION { color: #2ecc71; font-weight: bold; }
    .strategy-PREMIUM { color: #e74c3c; font-weight: bold; }
    .strategy-NEUTRAL { color: #f39c12; font-weight: bold; }
    .strategy-AGGRESSIVE { color: #9b59b6; font-weight: bold; }
    .strategy-CAUTIOUS { color: #3498db; font-weight: bold; }
    .strategy-HOLD { color: #95a5a6; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class PricingDashboard:
    """Classe principale du dashboard d'optimisation des prix."""
    
    def __init__(self):
        # Initialiser le state session si pas déjà fait
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.optimization_data = None
            st.session_state.product_features = None
            st.session_state.elasticity_data = None
            # Charger les données d'exemple au démarrage
            self._load_sample_data()
            st.session_state.data_loaded = True
        
        # Utiliser les données du session state
        self.data_loaded = st.session_state.data_loaded
        self.optimization_data = st.session_state.optimization_data
        self.product_features = st.session_state.product_features
        self.elasticity_data = st.session_state.elasticity_data
        
    def load_data(self, data_dir="../../outputs"):
        """Charge les données depuis les fichiers CSV."""
        try:
            # Vérifier si le dossier outputs existe
            if not os.path.exists(data_dir):
                st.error(f"❌ Dossier {data_dir} non trouvé")
                st.info("Exécutez d'abord les scripts d'optimisation pour générer les données")
                # Charger les données d'exemple à la place
                self._load_sample_data()
                self._save_to_session_state()
                return True
            
            # Charger les données d'optimisation
            opt_path = os.path.join(data_dir, "optimized_prices.csv")
            if os.path.exists(opt_path):
                self.optimization_data = pd.read_csv(opt_path)
                # Vérifier si la colonne 'success' existe
                if 'success' in self.optimization_data.columns:
                    self.optimization_data = self.optimization_data[self.optimization_data['success'] == True]
                st.sidebar.success(f"✅ {len(self.optimization_data)} optimisations chargées")
            else:
                st.sidebar.warning("📋 Fichier optimized_prices.csv non trouvé")
                self.optimization_data = None
            
            # Charger les features produits
            product_paths = [
                os.path.join(data_dir, "product_features.csv"),
                os.path.join(data_dir, "product_features_sample.csv"),
                "../../data/processed/master_features_sample.csv"
            ]
            
            for path in product_paths:
                if os.path.exists(path):
                    self.product_features = pd.read_csv(path)
                    st.sidebar.success(f"✅ {len(self.product_features)} produits chargés")
                    break
            else:
                st.sidebar.warning("📦 Aucun fichier de produits trouvé")
                self.product_features = None
            
            # Charger l'élasticité
            elasticity_path = os.path.join(data_dir, "elasticity_by_segment.csv")
            if os.path.exists(elasticity_path):
                self.elasticity_data = pd.read_csv(elasticity_path)
                st.sidebar.success(f"✅ {len(self.elasticity_data)} segments d'élasticité chargés")
            else:
                st.sidebar.info("📊 Fichier elasticity_by_segment.csv non trouvé")
                self.elasticity_data = None
            
            # Si pas d'optimisation mais des produits, générer des données simulées
            if self.optimization_data is None and self.product_features is not None:
                self._create_simulated_optimization_data()
            
            # Si toujours pas de données, charger les données d'exemple
            if self.optimization_data is None:
                st.sidebar.warning("📊 Utilisation des données d'exemple")
                self._load_sample_data()
            
            self.data_loaded = True
            self._save_to_session_state()
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Erreur chargement données: {e}")
            # Charger les données d'exemple
            self._load_sample_data()
            self.data_loaded = True
            self._save_to_session_state()
            st.sidebar.warning("📊 Utilisation des données d'exemple")
            return True
    
    def _save_to_session_state(self):
        """Sauvegarde les données dans le session state."""
        st.session_state.data_loaded = self.data_loaded
        st.session_state.optimization_data = self.optimization_data
        st.session_state.product_features = self.product_features
        st.session_state.elasticity_data = self.elasticity_data
    
    def _create_simulated_optimization_data(self):
        """Crée des données d'optimisation simulées à partir des produits."""
        st.sidebar.info("🔄 Génération de données d'optimisation simulées...")
        
        # Prendre un échantillon des produits
        sample_products = self.product_features.head(100).copy()
        
        # Générer des optimisations simulées
        np.random.seed(42)
        
        # Déterminer les colonnes disponibles
        price_col = 'avg_price' if 'avg_price' in sample_products.columns else 'price'
        category_col = 'product_category_name' if 'product_category_name' in sample_products.columns else 'category'
        
        self.optimization_data = pd.DataFrame({
            'product_id': sample_products['product_id'] if 'product_id' in sample_products.columns else [f'PROD_{i}' for i in range(len(sample_products))],
            'product_category_name': sample_products[category_col] if category_col in sample_products.columns else ['General'] * len(sample_products),
            'current_price': sample_products[price_col] if price_col in sample_products.columns else np.random.uniform(10, 500, len(sample_products)),
            'optimized_price': sample_products[price_col] * np.random.uniform(0.8, 1.4, len(sample_products)) if price_col in sample_products.columns else np.random.uniform(8, 600, len(sample_products)),
            'revenue_gain_pct': np.random.uniform(-5, 25, len(sample_products)),
            'price_change_pct': np.random.uniform(-20, 40, len(sample_products)),
            'elasticity': np.random.uniform(-2.5, -0.5, len(sample_products)),
            'strategy': np.random.choice(['PENETRATION', 'PREMIUM', 'NEUTRAL', 'CAUTIOUS'], len(sample_products)),
            'success': True
        })
        
        st.sidebar.success("✅ Données d'optimisation simulées générées")
    
    def _load_sample_data(self):
        """Charge des données d'exemple pour la démo."""
        st.sidebar.info("🎲 Chargement des données d'exemple...")
        
        n_products = 150
        np.random.seed(42)
        
        categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books', 'Beauty', 'Toys']
        
        # Données d'optimisation d'exemple
        self.optimization_data = pd.DataFrame({
            'product_id': [f'PROD_{i:03d}' for i in range(n_products)],
            'product_category_name': np.random.choice(categories, n_products),
            'current_price': np.random.uniform(10, 500, n_products),
            'optimized_price': np.random.uniform(8, 600, n_products),
            'revenue_gain_pct': np.random.uniform(-10, 30, n_products),
            'price_change_pct': np.random.uniform(-20, 40, n_products),
            'elasticity': np.random.uniform(-2.5, -0.3, n_products),
            'strategy': np.random.choice(['PENETRATION', 'PREMIUM', 'NEUTRAL', 'AGGRESSIVE', 'CAUTIOUS'], n_products),
            'success': True
        })
        
        # Features produits
        self.product_features = pd.DataFrame({
            'product_id': [f'PROD_{i:03d}' for i in range(n_products)],
            'product_category_name': np.random.choice(categories, n_products),
            'avg_price': np.random.uniform(10, 500, n_products),
            'total_orders': np.random.randint(10, 1000, n_products),
            'product_status': np.random.choice(['ACTIVE', 'DORMANT'], n_products)
        })
        
        # Élasticité par segment
        self.elasticity_data = pd.DataFrame({
            'product_category_name': categories,
            'mean_elasticity': [-1.8, -1.2, -0.8, -1.5, -0.6, -1.3, -1.1],
            'confidence_level': ['HIGH', 'MEDIUM', 'LOW', 'HIGH', 'MEDIUM', 'MEDIUM', 'LOW']
        })
    
    def display_overview_metrics(self):
        """Affiche les métriques globales."""
        st.markdown('<div class="main-header">📊 Tableau de Bord - Optimisation des Prix</div>', unsafe_allow_html=True)
        
        if self.optimization_data is None:
            st.warning("Aucune donnée d'optimisation disponible")
            return
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_gain = self.optimization_data['revenue_gain_pct'].mean()
            st.metric(
                "Gain de Revenue Moyen",
                f"{avg_gain:.1f}%",
                delta=f"{avg_gain:.1f}%"
            )
        
        with col2:
            total_products = len(self.optimization_data)
            st.metric("Produits Optimisés", f"{total_products:,}")
        
        with col3:
            positive_gains = len(self.optimization_data[self.optimization_data['revenue_gain_pct'] > 0])
            success_rate = (positive_gains / total_products) * 100
            st.metric("Taux de Succès", f"{success_rate:.1f}%")
        
        with col4:
            potential_revenue_gain = self.optimization_data['revenue_gain_pct'].sum() / 100
            st.metric("Gain Revenue Total", f"+{potential_revenue_gain:.1f}%")
        
        st.markdown("---")
        
        # Graphiques additionnels
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des gains de revenue
            fig = px.histogram(
                self.optimization_data,
                x='revenue_gain_pct',
                nbins=20,
                title="Distribution des Gains de Revenue (%)",
                labels={'revenue_gain_pct': 'Gain de Revenue (%)', 'count': 'Nombre de Produits'}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Seuil de rentabilité")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Prix actuel vs optimisé
            fig = px.scatter(
                self.optimization_data.sample(min(50, len(self.optimization_data))),
                x='current_price',
                y='optimized_price',
                color='revenue_gain_pct',
                size='price_change_pct',
                title="Prix Actuel vs Prix Optimisé",
                labels={'current_price': 'Prix Actuel (€)', 'optimized_price': 'Prix Optimisé (€)'}
            )
            # Ligne de référence (pas de changement)
            max_price = max(self.optimization_data['current_price'].max(), self.optimization_data['optimized_price'].max())
            fig.add_shape(type="line", x0=0, y0=0, x1=max_price, y1=max_price, 
                         line=dict(dash="dash", color="red"))
            st.plotly_chart(fig, use_container_width=True)
    
    def display_strategy_analysis(self):
        """Analyse des stratégies de pricing."""
        st.header("🎯 Analyse des Stratégies de Pricing")
        
        if self.optimization_data is None:
            st.warning("Aucune donnée d'optimisation disponible")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Distribution des stratégies
            strategy_counts = self.optimization_data['strategy'].value_counts()
            fig = px.pie(
                values=strategy_counts.values,
                names=strategy_counts.index,
                title="Répartition des Stratégies Recommandées",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Métriques par stratégie
            st.subheader("Performance par Stratégie")
            strategy_stats = self.optimization_data.groupby('strategy').agg({
                'revenue_gain_pct': 'mean',
                'price_change_pct': 'mean',
                'product_id': 'count'
            }).round(2)
            
            for strategy, stats in strategy_stats.iterrows():
                with st.container():
                    st.markdown(f'<div class="strategy-{strategy}">{strategy}</div>', unsafe_allow_html=True)
                    st.write(f"Gain moyen: {stats['revenue_gain_pct']}%")
                    st.write(f"Δ Prix: {stats['price_change_pct']}%")
                    st.write(f"Produits: {stats['product_id']}")
                    st.markdown("---")
        
        # Graphique performance par stratégie
        st.subheader("📊 Performance Détaillée par Stratégie")
        
        fig = px.box(
            self.optimization_data,
            x='strategy',
            y='revenue_gain_pct',
            color='strategy',
            title="Distribution des Gains par Stratégie",
            labels={'revenue_gain_pct': 'Gain de Revenue (%)', 'strategy': 'Stratégie'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    def display_elasticity_analysis(self):
        """Analyse de l'élasticité prix."""
        st.header("📈 Analyse d'Élasticité Prix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if self.optimization_data is not None and 'elasticity' in self.optimization_data.columns:
                # Relation élasticité vs gain de revenue
                fig = px.scatter(
                    self.optimization_data,
                    x='elasticity',
                    y='revenue_gain_pct',
                    color='strategy',
                    size='price_change_pct',
                    hover_data=['product_id'],
                    title="Élasticité vs Gain de Revenue",
                    labels={'elasticity': 'Élasticité Prix', 'revenue_gain_pct': 'Gain de Revenue (%)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.add_vline(x=-1, line_dash="dash", line_color="blue", annotation_text="Élastique = -1")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Données d'élasticité non disponibles")
        
        with col2:
            if self.elasticity_data is not None:
                # Élasticité par catégorie
                fig = px.bar(
                    self.elasticity_data.sort_values('mean_elasticity'),
                    x='mean_elasticity',
                    y='product_category_name',
                    orientation='h',
                    color='mean_elasticity',
                    title="Élasticité par Catégorie de Produits",
                    labels={'mean_elasticity': 'Élasticité Moyenne', 'product_category_name': 'Catégorie'}
                )
                fig.add_vline(x=-1, line_dash="dash", line_color="red", annotation_text="Élastique = -1")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Données d'élasticité par segment non disponibles")
        
        # Analyse supplémentaire si données disponibles
        if self.optimization_data is not None and 'elasticity' in self.optimization_data.columns:
            st.subheader("📊 Analyse Détaillée de l'Élasticité")
            
            # Histogramme des élasticités
            fig = px.histogram(
                self.optimization_data,
                x='elasticity',
                nbins=20,
                title="Distribution des Élasticités",
                labels={'elasticity': 'Élasticité Prix', 'count': 'Nombre de Produits'}
            )
            fig.add_vline(x=-1, line_dash="dash", line_color="red", annotation_text="Seuil élastique")
            st.plotly_chart(fig, use_container_width=True)
    
    def display_product_optimizations(self):
        """Détail des optimisations par produit."""
        st.header("📦 Optimisations par Produit")
        
        if self.optimization_data is None:
            st.warning("Aucune donnée d'optimisation disponible")
            return
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategies = st.multiselect(
                "Stratégies",
                options=self.optimization_data['strategy'].unique(),
                default=self.optimization_data['strategy'].unique()
            )
        
        with col2:
            min_gain = st.slider(
                "Gain de revenue minimum (%)",
                min_value=float(self.optimization_data['revenue_gain_pct'].min()),
                max_value=float(self.optimization_data['revenue_gain_pct'].max()),
                value=0.0,
                step=1.0
            )
        
        with col3:
            categories = st.multiselect(
                "Catégories",
                options=self.optimization_data['product_category_name'].unique(),
                default=self.optimization_data['product_category_name'].unique()
            )
        
        # Filtrage des données
        filtered_data = self.optimization_data[
            (self.optimization_data['strategy'].isin(strategies)) &
            (self.optimization_data['revenue_gain_pct'] >= min_gain) &
            (self.optimization_data['product_category_name'].isin(categories))
        ]
        
        # Affichage des résultats
        st.subheader(f"Résultats ({len(filtered_data)} produits)")
        
        if len(filtered_data) > 0:
            # Graphique des top gains
            top_products = filtered_data.nlargest(20, 'revenue_gain_pct')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_products['revenue_gain_pct'],
                y=top_products['product_id'],
                orientation='h',
                marker_color=top_products['revenue_gain_pct'].apply(
                    lambda x: 'green' if x > 0 else 'red'
                ),
                hovertemplate='<b>%{y}</b><br>Gain: %{x:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title="Top 20 des Produits par Gain de Revenue",
                xaxis_title="Gain de Revenue (%)",
                yaxis_title="Produit ID",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Table détaillée
            with st.expander("📋 Voir le détail des optimisations"):
                display_columns = [
                    'product_id', 'product_category_name', 'current_price', 
                    'optimized_price', 'price_change_pct', 'revenue_gain_pct', 'strategy'
                ]
                # Sélectionner seulement les colonnes disponibles
                available_columns = [col for col in display_columns if col in filtered_data.columns]
                st.dataframe(
                    filtered_data[available_columns].sort_values('revenue_gain_pct', ascending=False),
                    use_container_width=True
                )
        else:
            st.warning("Aucun produit ne correspond aux filtres sélectionnés")
    
    def display_what_if_analysis(self):
        """Analyse what-if pour tester différents scénarios."""
        st.header("🔮 Analyse What-If")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simulateur d'Élasticité")
            
            base_price = st.number_input("Prix actuel (€)", min_value=1.0, value=100.0, step=10.0)
            base_demand = st.number_input("Demande mensuelle", min_value=1, value=100, step=10)
            elasticity = st.slider("Élasticité prix", min_value=-3.0, max_value=0.0, value=-1.5, step=0.1)
            price_change = st.slider("Changement de prix (%)", min_value=-50, max_value=100, value=10, step=5)
            
            # Calcul des impacts
            new_price = base_price * (1 + price_change/100)
            demand_change = elasticity * (price_change/100)
            new_demand = base_demand * (1 + demand_change)
            
            current_revenue = base_price * base_demand
            new_revenue = new_price * new_demand
            revenue_change_pct = (new_revenue - current_revenue) / current_revenue * 100
            
        with col2:
            st.subheader("Résultats de la Simulation")
            
            # Métriques
            col21, col22 = st.columns(2)
            with col21:
                st.metric("Nouveau Prix", f"{new_price:.2f}€", f"{price_change}%")
                st.metric("Nouvelle Demande", f"{new_demand:.0f}", f"{demand_change*100:.1f}%")
            
            with col22:
                st.metric("Revenue Actuel", f"{current_revenue:.0f}€")
                st.metric("Nouveau Revenue", f"{new_revenue:.0f}€", f"{revenue_change_pct:.1f}%")
            
        # Visualisation sur toute la largeur
        st.subheader("📊 Courbe d'Optimisation du Prix")
        
        # Calculer les revenus pour différents prix
        price_changes = list(range(-30, 51, 2))
        prices = [base_price * (1 + p/100) for p in price_changes]
        revenues = []
        demands = []
        
        for p_change in price_changes:
            price = base_price * (1 + p_change/100)
            demand_change = elasticity * (p_change / 100)
            demand = base_demand * (1 + demand_change)
            revenue = price * demand
            revenues.append(revenue)
            demands.append(demand)
        
        # Créer le graphique
        fig = go.Figure()
        
        # Courbe de revenue
        fig.add_trace(go.Scatter(
            x=price_changes,
            y=revenues,
            mode='lines+markers',
            name='Revenue',
            line=dict(color='blue', width=3)
        ))
        
        # Point actuel
        fig.add_trace(go.Scatter(
            x=[0],
            y=[current_revenue],
            mode='markers',
            name='Prix Actuel',
            marker=dict(color='red', size=12)
        ))
        
        # Point simulé
        fig.add_trace(go.Scatter(
            x=[price_change],
            y=[new_revenue],
            mode='markers',
            name='Prix Simulé',
            marker=dict(color='green', size=12)
        ))
        
        fig.update_layout(
            title="Impact du Changement de Prix sur le Revenue",
            xaxis_title="Changement de Prix (%)",
            yaxis_title="Revenue (€)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations
        optimal_idx = np.argmax(revenues)
        optimal_price_change = price_changes[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        st.info(f"""
        📈 **Recommandation** : Le prix optimal serait un changement de **{optimal_price_change}%** 
        pour un revenue de **{optimal_revenue:.0f}€** (gain de **{((optimal_revenue-current_revenue)/current_revenue)*100:.1f}%**)
        """)
    
    def display_implementation_plan(self):
        """Plan d'implémentation des recommandations."""
        st.header("🚀 Plan d'Implémentation")
        
        if self.optimization_data is None:
            st.warning("Aucune donnée d'optimisation disponible")
            return
        
        # Priorisation des actions
        implementation_plan = self.optimization_data.copy()
        implementation_plan['priority'] = implementation_plan['revenue_gain_pct'].apply(
            lambda x: 'HIGH' if x > 15 else 'MEDIUM' if x > 5 else 'LOW'
        )
        
        # Par stratégie
        strategy_plan = implementation_plan.groupby(['strategy', 'priority']).agg({
            'product_id': 'count',
            'revenue_gain_pct': 'mean',
            'price_change_pct': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Plan par Stratégie")
            for strategy in strategy_plan['strategy'].unique():
                strategy_data = strategy_plan[strategy_plan['strategy'] == strategy]
                
                with st.expander(f"Stratégie {strategy} ({len(strategy_data)} priorités)"):
                    for _, row in strategy_data.iterrows():
                        priority_color = {
                            'HIGH': '🔴',
                            'MEDIUM': '🟡', 
                            'LOW': '🟢'
                        }.get(row['priority'], '⚪')
                        
                        st.write(f"{priority_color} **{row['priority']}** - {row['product_id']} produits")
                        st.write(f"Gain moyen: {row['revenue_gain_pct']:.1f}%")
                        st.write(f"Δ Prix moyen: {row['price_change_pct']:.1f}%")
                        st.markdown("---")
        
        with col2:
            st.subheader("Recommandations d'Implémentation")
            
            recommendations = {
                'PENETRATION': "✅ Implémenter rapidement - impact volume positif",
                'PREMIUM': "⚠️ Tester sur segment cible avant déploiement large",
                'NEUTRAL': "👀 Surveiller la concurrence - ajustements mineurs",
                'AGGRESSIVE': "🔍 Évaluer les risques concurrentiels",
                'CAUTIOUS': "⏳ Attendre plus de données de marché",
                'HOLD': "⏸️ Maintenir le statu quo - surveiller les indicateurs"
            }
            
            for strategy, recommendation in recommendations.items():
                if strategy in implementation_plan['strategy'].values:
                    st.markdown(f"**{strategy}**: {recommendation}")
            
        # Graphique de priorisation
        st.subheader("📊 Matrice de Priorisation")
        
        fig = px.scatter(
            implementation_plan,
            x='price_change_pct',
            y='revenue_gain_pct',
            color='priority',
            size='current_price',
            hover_data=['product_id', 'strategy'],
            title="Matrice Risque vs Rendement",
            labels={
                'price_change_pct': 'Changement de Prix (%)',
                'revenue_gain_pct': 'Gain de Revenue (%)',
                'priority': 'Priorité'
            }
        )
        
        fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Seuil Medium")
        fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Seuil High")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline d'implémentation
        st.subheader("📅 Timeline d'Implémentation Suggérée")
        
        priority_order = ['HIGH', 'MEDIUM', 'LOW']
        timeline_data = []
        
        for i, priority in enumerate(priority_order):
            priority_products = implementation_plan[implementation_plan['priority'] == priority]
            if len(priority_products) > 0:
                timeline_data.append({
                    'Phase': f"Phase {i+1}",
                    'Priorité': priority,
                    'Nombre de Produits': len(priority_products),
                    'Gain Moyen (%)': priority_products['revenue_gain_pct'].mean(),
                    'Semaines': f"{i*2+1}-{(i+1)*2}"
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True)
        
    def run(self):
        """Exécute le dashboard complet."""
        # Sidebar
        st.sidebar.title("⚙️ Configuration")
        
        st.sidebar.markdown("""
        ### 📁 Source des données
        Le dashboard lit les fichiers CSV générés par les scripts d'optimisation.
        """)
        
        # Status des données
        if self.data_loaded:
            st.sidebar.success("✅ Données chargées avec succès")
            if self.optimization_data is not None:
                st.sidebar.info(f"📊 {len(self.optimization_data)} optimisations disponibles")
        else:
            st.sidebar.warning("⚠️ Données non chargées")
        
        # Bouton de chargement
        if st.sidebar.button("🔄 Charger les données depuis outputs/"):
            with st.spinner("Chargement des données..."):
                success = self.load_data("../../outputs")
                if success:
                    st.sidebar.success("✅ Données rechargées avec succès")
                    st.rerun()
        
        # Instructions
        st.sidebar.markdown("---")
        st.sidebar.title("📋 Instructions")
        
        with st.sidebar.expander("📖 Guide d'utilisation"):
            st.markdown("""
            **Pour générer les vraies données :**
            1. `python load_kaggle_olist.py`
            2. `python transform.py`  
            3. `python price_optimizer.py`
            4. Cliquez sur "Charger les données"
            
            **Navigation :**
            - Utilisez le menu radio pour naviguer
            - Chaque section offre des analyses différentes
            - Les filtres sont interactifs
            """)
        
        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.title("📊 Navigation")
        
        sections = {
            "📊 Vue d'ensemble": self.display_overview_metrics,
            "🎯 Stratégies Pricing": self.display_strategy_analysis,
            "📈 Analyse Élasticité": self.display_elasticity_analysis,
            "📦 Optimisations Produits": self.display_product_optimizations,
            "🔮 Simulation What-If": self.display_what_if_analysis,
            "🚀 Plan d'Implémentation": self.display_implementation_plan
        }
        
        selected_section = st.sidebar.radio("Sélectionnez une section", list(sections.keys()))
        
        # Affichage de la section sélectionnée
        try:
            sections[selected_section]()
        except Exception as e:
            st.error(f"Erreur lors de l'affichage de la section: {str(e)}")
            st.info("Essayez de recharger les données ou contactez l'administrateur")
    
    def display_setup_instructions(self):
        """Affiche les instructions de setup."""
        st.markdown('<div class="main-header">🚀 Dashboard Optimisation Prix E-commerce</div>', unsafe_allow_html=True)
        
        st.info("""
        ### 📋 Pré-requis pour utiliser le dashboard
        
        Pour avoir des données complètes, exécutez dans l'ordre :
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Ingestion des données")
            st.code("""
            python load_kaggle_olist.py \\
                --csv-path data/raw \\
                --pg-uri postgresql://user:pass@localhost:5432/dbname
            """)
        
        with col2:
            st.subheader("2. Transformations")
            st.code("""
            python transform.py \\
                --pg-uri postgresql://user:pass@localhost:5432/dbname
            """)
        
        with col3:
            st.subheader("3. Optimisation")
            st.code("""
            python price_optimizer.py \\
                --pg-uri postgresql://postgres:root@localhost:5432/opt_db \\
                --max-products 100
            """)
        
        st.markdown("""
        ### 🎯 Fonctionnalités disponibles
        
        - **📊 Vue d'ensemble** : Métriques clés et KPI
        - **🎯 Stratégies Pricing** : Analyse des recommandations par stratégie  
        - **📈 Analyse Élasticité** : Impact de l'élasticité prix sur les revenus
        - **📦 Optimisations Produits** : Détail des optimisations par produit
        - **🔮 Simulation What-If** : Test de différents scénarios de pricing
        - **🚀 Plan d'Implémentation** : Feuille de route pour le déploiement
        
        ### 🚀 Pour commencer :
        1. Le dashboard utilise des données d'exemple par défaut
        2. Naviguez entre les sections avec le menu de gauche
        3. Pour vos vraies données, exécutez les scripts et rechargez
        """)

def main():
    """Fonction principale."""
    dashboard = PricingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()