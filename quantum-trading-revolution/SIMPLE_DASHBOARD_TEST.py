"""
🚀 SIMPLE DASHBOARD TEST - Version minimale

Test rapide pour vérifier Dash + Plotly
"""

import dash
from dash import html, dcc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test simple des imports
print("🧪 TESTING IMPORTS...")
print("✅ dash imported")
print("✅ plotly imported") 
print("✅ pandas imported")
print("✅ numpy imported")

# Création app simple
app = dash.Dash(__name__)

# Données de test
dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
values = [100 + i * 1.65 for i in range(10)]  # Simulation 16.5% growth

# Graphique simple
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name='Portfolio'))
fig.update_layout(title="📈 Agent Portfolio Test", template='plotly_dark')

# Layout minimal
app.layout = html.Div([
    html.H1("🚀 Agent Dashboard - Test Version", style={'color': '#00D4AA'}),
    html.P("Status: Testing Interface", style={'color': 'white'}),
    dcc.Graph(figure=fig),
    html.P("✅ Si vous voyez ce graphique, l'interface fonctionne!", style={'color': 'green'})
], style={'backgroundColor': '#1e1e1e', 'padding': 20})

if __name__ == '__main__':
    print("\n🚀 LAUNCHING SIMPLE TEST...")
    print("🌐 Access: http://localhost:8051")  # Port différent
    print("🔧 Press Ctrl+C to stop")
    
    try:
        app.run(debug=False, port=8051, host='127.0.0.1')
    except Exception as e:
        print(f"❌ ERROR: {e}")
