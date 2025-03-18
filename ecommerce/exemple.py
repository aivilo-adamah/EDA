import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from shiny import App, ui, render

# Charger les données (Remplace par ton dataset)
df = pd.read_csv("data/data.csv", encoding="latin1")  # Remplace par ton fichier

# Pré-traitement des données
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(str)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Interface Utilisateur
app_ui = ui.page_fluid(
    ui.h2("Analyse des ventes 📊"),
    
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("country", "Sélectionnez un pays", 
                            choices=["Tous"] + list(df['Country'].unique()), 
                            selected="Tous"),
            ui.input_slider("top_n", "Nombre de produits affichés", min=5, max=20, value=10),
            ui.input_action_button("refresh", "Actualiser 🔄")
        ),
        
        ui.card_body(
            ui.output_table("data_summary"),
            ui.output_plot("sales_trend"),
            ui.output_plot("top_products"),
            ui.output_plot("top_clients"),
            ui.output_plot("rfm_segments"),
        )
    )
)

# Serveur
def server(input, output, session):
    @output
    @render.table
    def data_summary():
        """Aperçu des données"""
        return df.describe()

    @output
    @render.plot
    def sales_trend():
        """Évolution des ventes mensuelles"""
        df_monthly = df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum()
        plt.figure(figsize=(12, 5))
        plt.plot(df_monthly, marker='o', linestyle='-', color='b')
        plt.title("Évolution du chiffre d'affaires mensuel")
        plt.xlabel("Date")
        plt.ylabel("CA (€)")
        plt.grid()
        return plt.gcf()

    @output
    @render.plot
    def top_products():
        """Top produits les plus vendus"""
        top_n = input.top_n()
        top_products = df['Description'].value_counts().head(top_n)
        plt.figure(figsize=(12, 5))
        sns.barplot(y=top_products.index, x=top_products.values, palette="viridis")
        plt.title(f"Top {top_n} produits les plus vendus")
        plt.xlabel("Nombre de ventes")
        plt.ylabel("Produits")
        return plt.gcf()

    @output
    @render.plot
    def top_clients():
        """Top clients les plus rentables"""
        top_clients = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 5))
        sns.barplot(y=top_clients.index, x=top_clients.values, palette="magma")
        plt.title("Top 10 clients les plus rentables")
        plt.xlabel("Revenu total (€)")
        plt.ylabel("Clients")
        return plt.gcf()

    @output
    @render.plot
    def rfm_segments():
        """Segmentation RFM"""
        date_ref = df['InvoiceDate'].max()
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (date_ref - x.max()).days,  # Récence
            'InvoiceNo': 'count',  # Fréquence
            'TotalPrice': 'sum'  # Monétaire
        })
        rfm.rename(columns={'InvoiceDate': 'Recence', 'InvoiceNo': 'Frequence', 'TotalPrice': 'Monetaire'}, inplace=True)
        rfm['R_Score'] = pd.qcut(rfm['Recence'], q=4, labels=[4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequence'], q=4, labels=[1, 2, 3, 4])
        rfm['M_Score'] = pd.qcut(rfm['Monetaire'], q=4, labels=[1, 2, 3, 4])
        rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int).sum(axis=1)
        rfm['Segment'] = rfm['RFM_Score'].apply(lambda x: 'VIP' if x >= 10 else 'Fidèle' if x >= 7 else 'Occasionnel')

        plt.figure(figsize=(8, 5))
        sns.countplot(x=rfm['Segment'], palette="coolwarm")
        plt.title("Répartition des clients par segment")
        plt.xlabel("Segment")
        plt.ylabel("Nombre de clients")
        return plt.gcf()

# Lancer l'application
app = App(app_ui, server)

