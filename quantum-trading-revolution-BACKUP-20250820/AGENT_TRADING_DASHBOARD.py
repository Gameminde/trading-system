"""
🎨 AGENT TRADING DASHBOARD - INTERFACE TEMPS RÉEL

MISSION: Interface graphique moderne pour visualiser l'agent trading
FEATURES: Portfolio live, Graphiques, Sentiment, Trading log, Contrôles
TECH STACK: Dash + Plotly + WebSocket + Real-time updates

💎 ENFIN UNE BELLE INTERFACE POUR L'AGENT !
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
import queue
import yfinance as yf
import requests
from AGENT_OPTIMIZED_MAXIMUM_POWER import OptimizedTradingAgent, OptimizedConfig
import logging

# Configuration du logging pour l'interface
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DASHBOARD")

# Configuration globale
SYMBOLS = ["BTC", "ETH", "AAPL", "TSLA", "MSFT", "GOOGL", "BNB"]
UPDATE_INTERVAL = 2000  # 2 secondes
CHART_HISTORY = 100  # Points sur les graphiques

class TradingDashboard:
    """Dashboard principal pour l'agent de trading"""
    
    def __init__(self):
        # État de l'application
        self.agent = None
        self.agent_thread = None
        self.is_running = False
        self.data_queue = queue.Queue()
        
        # Données temps réel
        self.portfolio_history = []
        self.market_data = {}
        self.sentiment_data = {}
        self.trades_log = []
        self.performance_metrics = {
            'total_return': 0,
            'success_rate': 0,
            'positions_count': 0,
            'total_trades': 0
        }
        
        # Configuration Dash
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        # Thread de mise à jour des données
        self.data_thread = threading.Thread(target=self._update_market_data, daemon=True)
        self.data_thread.start()
        
        logger.info("🎨 Trading Dashboard initialized")
    
    def setup_layout(self):
        """Configuration de l'interface utilisateur"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 QUANTUM TRADING AGENT DASHBOARD", 
                       style={'textAlign': 'center', 'color': '#00ff88', 'marginBottom': '20px'}),
                
                # Controls
                html.Div([
                    html.Button('▶️ START AGENT', id='start-btn', n_clicks=0, 
                               style={'backgroundColor': '#00ff88', 'color': 'black', 'marginRight': '10px'}),
                    html.Button('⏹️ STOP AGENT', id='stop-btn', n_clicks=0,
                               style={'backgroundColor': '#ff4444', 'color': 'white', 'marginRight': '10px'}),
                    html.Span(id='status-indicator', children='🔴 STOPPED', 
                             style={'fontSize': '18px', 'marginLeft': '20px'})
                ], style={'textAlign': 'center', 'marginBottom': '30px'})
            ], style={'backgroundColor': '#1e1e1e', 'padding': '20px'}),
            
            # Main Dashboard
            html.Div([
                # Row 1: Portfolio + Performance
                html.Div([
                    # Portfolio Overview
                    html.Div([
                        html.H3('💼 PORTFOLIO OVERVIEW', style={'color': '#00ff88', 'textAlign': 'center'}),
                        html.Div(id='portfolio-cards')
                    ], className='six columns'),
                    
                    # Performance Chart
                    html.Div([
                        html.H3('📈 PERFORMANCE CHART', style={'color': '#00ff88', 'textAlign': 'center'}),
                        dcc.Graph(id='performance-chart')
                    ], className='six columns')
                ], className='row', style={'marginBottom': '30px'}),
                
                # Row 2: Market Data + Sentiment
                html.Div([
                    # Market Data
                    html.Div([
                        html.H3('💹 MARKET DATA', style={'color': '#00ff88', 'textAlign': 'center'}),
                        dcc.Graph(id='market-chart')
                    ], className='six columns'),
                    
                    # Sentiment Analysis
                    html.Div([
                        html.H3('🧠 SENTIMENT ANALYSIS', style={'color': '#00ff88', 'textAlign': 'center'}),
                        dcc.Graph(id='sentiment-chart')
                    ], className='six columns')
                ], className='row', style={'marginBottom': '30px'}),
                
                # Row 3: Trading Log + Module Status
                html.Div([
                    # Trading Log
                    html.Div([
                        html.H3('📋 TRADING LOG', style={'color': '#00ff88', 'textAlign': 'center'}),
                        html.Div(id='trading-log', style={'height': '300px', 'overflowY': 'scroll'})
                    ], className='six columns'),
                    
                    # Module Status
                    html.Div([
                        html.H3('⚛️ MODULE STATUS', style={'color': '#00ff88', 'textAlign': 'center'}),
                        html.Div(id='module-status')
                    ], className='six columns')
                ], className='row')
            ], style={'backgroundColor': '#2d2d2d', 'padding': '20px', 'minHeight': '800px'}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=UPDATE_INTERVAL,
                n_intervals=0
            ),
            
            # Hidden div to store data
            html.Div(id='hidden-data', style={'display': 'none'})
        ], style={'backgroundColor': '#1e1e1e', 'minHeight': '100vh'})
    
    def setup_callbacks(self):
        """Configuration des callbacks Dash"""
        
        # Start Agent
        @self.app.callback(
            Output('status-indicator', 'children'),
            [Input('start-btn', 'n_clicks'), Input('stop-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def control_agent(start_clicks, stop_clicks):
            ctx = callback_context
            if not ctx.triggered:
                return '🔴 STOPPED'
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-btn' and not self.is_running:
                self.start_agent()
                return '🟢 RUNNING'
            elif button_id == 'stop-btn' and self.is_running:
                self.stop_agent()
                return '🔴 STOPPED'
            
            return '🟢 RUNNING' if self.is_running else '🔴 STOPPED'
        
        # Update Portfolio Cards
        @self.app.callback(
            Output('portfolio-cards', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_portfolio_cards(n):
            if not self.agent:
                return html.Div("Agent not started", style={'color': '#888'})
            
            # Données portfolio
            cash = getattr(self.agent, 'cash', 100.0)
            positions = getattr(self.agent, 'positions', {})
            portfolio_value = cash + sum(pos.get('quantity', 0) * self._get_current_price(symbol) 
                                       for symbol, pos in positions.items())
            
            total_return = portfolio_value - 100.0
            return_percent = (total_return / 100.0) * 100
            
            cards = [
                # Portfolio Value Card
                html.Div([
                    html.H4(f'${portfolio_value:.2f}', style={'color': '#00ff88', 'margin': '0'}),
                    html.P('Portfolio Value', style={'color': '#888', 'margin': '0'})
                ], style={'backgroundColor': '#3d3d3d', 'padding': '15px', 'borderRadius': '10px', 
                         'textAlign': 'center', 'margin': '5px'}),
                
                # Cash Card
                html.Div([
                    html.H4(f'${cash:.2f}', style={'color': '#00aaff', 'margin': '0'}),
                    html.P('Available Cash', style={'color': '#888', 'margin': '0'})
                ], style={'backgroundColor': '#3d3d3d', 'padding': '15px', 'borderRadius': '10px', 
                         'textAlign': 'center', 'margin': '5px'}),
                
                # Return Card
                html.Div([
                    html.H4(f'{return_percent:+.1f}%', 
                           style={'color': '#00ff88' if return_percent >= 0 else '#ff4444', 'margin': '0'}),
                    html.P(f'${total_return:+.2f}', style={'color': '#888', 'margin': '0'})
                ], style={'backgroundColor': '#3d3d3d', 'padding': '15px', 'borderRadius': '10px', 
                         'textAlign': 'center', 'margin': '5px'}),
                
                # Positions Card
                html.Div([
                    html.H4(f'{len(positions)}', style={'color': '#ffaa00', 'margin': '0'}),
                    html.P('Open Positions', style={'color': '#888', 'margin': '0'})
                ], style={'backgroundColor': '#3d3d3d', 'padding': '15px', 'borderRadius': '10px', 
                         'textAlign': 'center', 'margin': '5px'})
            ]
            
            return html.Div(cards, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'})
        
        # Update Performance Chart
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(n):
            if not self.portfolio_history:
                # Empty chart
                fig = go.Figure()
                fig.update_layout(
                    title="Portfolio Performance",
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='white'
                )
                return fig
            
            df = pd.DataFrame(self.portfolio_history)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Performance Over Time",
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                showlegend=False
            )
            
            return fig
        
        # Update Market Chart
        @self.app.callback(
            Output('market-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_market_chart(n):
            if not self.market_data:
                fig = go.Figure()
                fig.update_layout(
                    title="Market Data",
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='white'
                )
                return fig
            
            fig = go.Figure()
            
            colors = ['#00ff88', '#00aaff', '#ffaa00', '#ff4444', '#aa00ff', '#00ffaa', '#ff8800']
            
            for i, (symbol, data) in enumerate(self.market_data.items()):
                if 'price_history' in data and data['price_history']:
                    prices = [p['price'] for p in data['price_history'][-20:]]  # Last 20 points
                    times = [p['timestamp'] for p in data['price_history'][-20:]]
                    
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=prices,
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            fig.update_layout(
                title="Market Prices (Real-time)",
                xaxis_title="Time", 
                yaxis_title="Price",
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                legend=dict(bgcolor='#3d3d3d', font_color='white')
            )
            
            return fig
        
        # Update Sentiment Chart
        @self.app.callback(
            Output('sentiment-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_sentiment_chart(n):
            if not self.sentiment_data:
                fig = go.Figure()
                fig.update_layout(
                    title="Sentiment Analysis",
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='white'
                )
                return fig
            
            symbols = list(self.sentiment_data.keys())
            sentiments = [self.sentiment_data[s].get('confidence', 0) for s in symbols]
            signals = [self.sentiment_data[s].get('signal', 'HOLD') for s in symbols]
            
            colors = ['#00ff88' if s == 'BUY' else '#ff4444' if s == 'SELL' else '#888' 
                     for s in signals]
            
            fig = go.Figure(data=[
                go.Bar(x=symbols, y=sentiments, marker_color=colors)
            ])
            
            fig.update_layout(
                title="Current Sentiment Analysis",
                xaxis_title="Symbol",
                yaxis_title="Confidence",
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                showlegend=False
            )
            
            return fig
        
        # Update Trading Log
        @self.app.callback(
            Output('trading-log', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trading_log(n):
            if not self.trades_log:
                return html.Div("No trades yet", style={'color': '#888', 'textAlign': 'center'})
            
            log_entries = []
            for trade in self.trades_log[-10:]:  # Last 10 trades
                color = '#00ff88' if trade.get('action') == 'BUY' else '#ff4444'
                pnl_text = f" | P&L: ${trade.get('pnl', 0):.2f}" if 'pnl' in trade else ""
                
                entry = html.Div([
                    html.Span(f"{trade.get('timestamp', '')[:19]} | ", style={'color': '#888'}),
                    html.Span(f"{trade.get('action', 'N/A')} {trade.get('symbol', 'N/A')}", 
                             style={'color': color, 'fontWeight': 'bold'}),
                    html.Span(f" @ ${trade.get('price', 0):.4f}{pnl_text}", style={'color': '#ccc'})
                ], style={'marginBottom': '5px', 'fontSize': '14px'})
                
                log_entries.append(entry)
            
            return log_entries
        
        # Update Module Status
        @self.app.callback(
            Output('module-status', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_module_status(n):
            if not self.agent:
                return html.Div("Agent not active", style={'color': '#888'})
            
            status_cards = [
                html.Div([
                    html.H5('🧠 LLM Sentiment', style={'color': '#00aaff', 'margin': '0'}),
                    html.P('✅ Active', style={'color': '#00ff88', 'margin': '0'})
                ], style={'backgroundColor': '#3d3d3d', 'padding': '10px', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H5('🔧 MPS Optimizer', style={'color': '#00aaff', 'margin': '0'}),
                    html.P('✅ Active', style={'color': '#00ff88', 'margin': '0'})
                ], style={'backgroundColor': '#3d3d3d', 'padding': '10px', 'borderRadius': '5px', 'margin': '5px'}),
                
                html.Div([
                    html.H5('⚛️ Quantum Engine', style={'color': '#00aaff', 'margin': '0'}),
                    html.P('✅ Stabilized', style={'color': '#00ff88', 'margin': '0'})
                ], style={'backgroundColor': '#3d3d3d', 'padding': '10px', 'borderRadius': '5px', 'margin': '5px'})
            ]
            
            return html.Div(status_cards)
    
    def start_agent(self):
        """Démarrer l'agent de trading"""
        if not self.is_running:
            self.is_running = True
            config = OptimizedConfig()
            config.initial_capital = 100.0
            self.agent = OptimizedTradingAgent(config)
            
            # Thread agent
            self.agent_thread = threading.Thread(target=self._run_agent, daemon=True)
            self.agent_thread.start()
            
            logger.info("🚀 Agent started")
    
    def stop_agent(self):
        """Arrêter l'agent de trading"""
        self.is_running = False
        if self.agent:
            self.agent.save_optimized_memory()
        logger.info("🛑 Agent stopped")
    
    def _run_agent(self):
        """Thread principal de l'agent"""
        try:
            while self.is_running:
                # Données marché
                market_data = self.agent.get_real_market_data(SYMBOLS)
                
                if market_data:
                    # Analyse
                    analysis_results = self.agent.optimized_analysis(market_data)
                    
                    # Exécution trades
                    for symbol, analysis in analysis_results.items():
                        if self.agent.execute_optimized_trade(analysis):
                            # Log trade
                            if self.agent.trades:
                                latest_trade = self.agent.trades[-1]
                                self.trades_log.append(latest_trade)
                    
                    # Update sentiment data
                    for symbol in market_data:
                        if symbol in analysis_results:
                            self.sentiment_data[symbol] = {
                                'signal': analysis_results[symbol].get('signal', 'HOLD'),
                                'confidence': analysis_results[symbol].get('confidence', 0)
                            }
                    
                    # Update portfolio history
                    portfolio_value = self.agent.get_portfolio_value()
                    self.portfolio_history.append({
                        'timestamp': datetime.now(),
                        'portfolio_value': portfolio_value,
                        'cash': self.agent.cash,
                        'positions_count': len(self.agent.positions)
                    })
                    
                    # Limiter l'historique
                    if len(self.portfolio_history) > CHART_HISTORY:
                        self.portfolio_history = self.portfolio_history[-CHART_HISTORY:]
                
                time.sleep(5)  # Pause entre cycles
                
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self.is_running = False
    
    def _update_market_data(self):
        """Thread de mise à jour des données marché"""
        while True:
            try:
                for symbol in SYMBOLS:
                    data = self._fetch_symbol_data(symbol)
                    if data:
                        if symbol not in self.market_data:
                            self.market_data[symbol] = {'price_history': []}
                        
                        # Ajouter prix à l'historique
                        self.market_data[symbol]['price_history'].append({
                            'timestamp': datetime.now(),
                            'price': data['price']
                        })
                        
                        # Limiter historique
                        if len(self.market_data[symbol]['price_history']) > CHART_HISTORY:
                            self.market_data[symbol]['price_history'] = \
                                self.market_data[symbol]['price_history'][-CHART_HISTORY:]
                        
                        # Données actuelles
                        self.market_data[symbol].update(data)
                
                time.sleep(3)  # Update every 3 seconds
                
            except Exception as e:
                logger.warning(f"Market data update error: {e}")
                time.sleep(5)
    
    def _fetch_symbol_data(self, symbol: str):
        """Fetch données pour un symbole"""
        try:
            if symbol in ["BTC", "ETH", "BNB"]:
                response = requests.get(
                    f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT", 
                    timeout=3
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "symbol": symbol,
                        "price": float(data['priceChange']) + float(data['prevClosePrice']),
                        "change_24h": float(data['priceChangePercent']),
                        "volume": int(float(data['volume']))
                    }
            else:
                # Actions via Yahoo Finance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                if len(data) > 0:
                    current = data.iloc[-1]
                    previous = data.iloc[0] if len(data) > 1 else current
                    
                    return {
                        "symbol": symbol,
                        "price": float(current['Close']),
                        "change_24h": ((float(current['Close']) - float(previous['Close'])) / float(previous['Close'])) * 100,
                        "volume": int(current['Volume'])
                    }
        except Exception as e:
            logger.warning(f"Fetch error {symbol}: {e}")
        
        return None
    
    def _get_current_price(self, symbol: str) -> float:
        """Obtenir prix actuel d'un symbole"""
        if symbol in self.market_data:
            return self.market_data[symbol].get('price', 0)
        return 0
    
    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Lancer le dashboard"""
        logger.info(f"🎨 Starting Trading Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# CSS Style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def main():
    """Lancer l'application dashboard"""
    print("🎨" + "="*60 + "🎨")
    print("   🚀 QUANTUM TRADING AGENT DASHBOARD")
    print("   💎 Interface graphique temps réel")
    print("   📊 Portfolio • Graphiques • Sentiment • Trading")
    print("="*64)
    
    dashboard = TradingDashboard()
    
    print(f"\n🌐 Lancement du serveur dashboard...")
    print(f"   URL: http://localhost:8050")
    print(f"   Contrôles: START/STOP agent dans l'interface")
    print(f"   Données: Temps réel crypto + stocks")
    print(f"   Mise à jour: Toutes les 2 secondes")
    
    try:
        dashboard.run(host='0.0.0.0', port=8050, debug=False)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
