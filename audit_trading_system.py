"""
🔍 AUDIT COMPLET DU SYSTÈME DE TRADING
Analyse de la rentabilité réelle et de la capacité d'apprentissage
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AUDIT')

def run_complete_audit():
    print('='*70)
    print('🔍 AUDIT COMPLET DU SYSTÈME DE TRADING')
    print('='*70)
    
    # Test 1: Vérifier le capital initial
    print('\n📊 TEST 1: Configuration du Capital')
    print('-'*40)
    from rl_trading_agent import TradingEnvironment
    
    # Créer environnement avec capital par défaut
    test_data = pd.DataFrame({'close': [100], 'rsi': [50], 'macd': [0], 'volume': [1000000]})
    env = TradingEnvironment(test_data)
    print(f'Capital initial actuel: ${env.initial_balance:,.2f}')
    print(f'Balance initiale: ${env.balance:,.2f}')
    
    # Test avec capital de 1000$
    env_1000 = TradingEnvironment(test_data, initial_balance=1000)
    print(f'Test avec capital de 1000$: ${env_1000.initial_balance:,.2f}')
    
    # Test 2: Vérifier la capacité d'apprentissage
    print('\n🧠 TEST 2: Capacité d\'Apprentissage')
    print('-'*40)
    
    # Créer données de test avec tendance claire
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    trend_data = pd.DataFrame({
        'close': 100 + np.arange(len(dates)) * 0.1,  # Tendance haussière
        'rsi': 50 + np.sin(np.arange(len(dates)) * 0.1) * 20,
        'macd': np.sin(np.arange(len(dates)) * 0.05) * 2,
        'volume': np.random.uniform(900000, 1100000, len(dates))
    }, index=dates)
    
    from rl_trading_agent import RLTrader
    
    rl_trader = RLTrader()
    print('Entraînement du modèle RL sur données avec tendance...')
    
    # Entraîner sur première moitié
    train_data = trend_data.iloc[:180]
    rl_trader.train_model(train_data, total_timesteps=2000)
    
    # Évaluer sur deuxième moitié
    test_data = trend_data.iloc[180:]
    metrics = rl_trader.evaluate_performance(test_data)
    
    print(f'Résultats après entraînement:')
    print(f'  - Return final: {metrics.get("final_return", 0):.2f}%')
    print(f'  - Win rate: {metrics.get("win_rate", 0):.1%}')
    print(f'  - Nombre de trades: {metrics.get("total_trades", 0)}')
    
    # Test 3: Vérifier la rentabilité réelle
    print('\n💰 TEST 3: Rentabilité Réelle')
    print('-'*40)
    
    # Simuler trading avec capital de 1000$
    env_real = TradingEnvironment(test_data, initial_balance=1000)
    obs, _ = env_real.reset()
    done = False
    trades_log = []
    
    while not done:
        action, _ = rl_trader.model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_real.step(action)
        
        if 'action' in info:
            trades_log.append(info)
    
    final_balance = env_real.balance
    profit = final_balance - 1000
    roi = (profit / 1000) * 100
    
    print(f'Capital initial: $1,000.00')
    print(f'Balance finale: ${final_balance:.2f}')
    print(f'Profit/Perte: ${profit:.2f}')
    print(f'ROI: {roi:.2f}%')
    
    if len(trades_log) > 0:
        winning_trades = [t for t in trades_log if t.get('profit', 0) > 0]
        print(f'Trades gagnants: {len(winning_trades)}/{len(trades_log)}')
    
    # Test 4: Analyse de la mémoire et apprentissage continu
    print('\n🔄 TEST 4: Mémoire et Apprentissage Continu')
    print('-'*40)
    
    # Vérifier si le modèle s'améliore avec le temps
    performance_history = []
    
    for epoch in range(3):
        print(f'\nÉpoque {epoch + 1}/3:')
        # Générer nouvelles données
        new_dates = pd.date_range(start=f'2024-0{epoch+1}-01', end=f'2024-0{epoch+1}-30', freq='D')
        new_data = pd.DataFrame({
            'close': 100 + np.random.randn(len(new_dates)) * 2,
            'rsi': 50 + np.random.randn(len(new_dates)) * 10,
            'macd': np.random.randn(len(new_dates)) * 0.5,
            'volume': np.random.uniform(900000, 1100000, len(new_dates))
        }, index=new_dates)
        
        # Continuer l'entraînement
        rl_trader.train_model(new_data, total_timesteps=500)
        
        # Évaluer performance
        eval_metrics = rl_trader.evaluate_performance(new_data)
        performance_history.append(eval_metrics.get('final_return', 0))
        print(f'  Performance: {eval_metrics.get("final_return", 0):.2f}%')
    
    # Vérifier amélioration
    if len(performance_history) > 1:
        improvement = performance_history[-1] - performance_history[0]
        print(f'\n📈 Amélioration sur 3 époques: {improvement:.2f}%')
        
        if improvement > 0:
            print('✅ Le modèle montre une capacité d\'apprentissage positive')
        else:
            print('⚠️ Le modèle ne montre pas d\'amélioration significative')
    
    # Test 5: Analyse des problèmes critiques
    print('\n⚠️ TEST 5: Problèmes Critiques Identifiés')
    print('-'*40)
    
    issues = []
    
    # Vérifier le capital
    if env.initial_balance == 100000:
        issues.append('Capital initial trop élevé (100k$) - peu réaliste pour tests')
    
    # Vérifier les données
    if env.shares == 0 and env.balance == env.initial_balance:
        issues.append('Agent ne fait peut-être pas de trades réels')
    
    # Vérifier la rentabilité
    if roi < 0:
        issues.append(f'ROI négatif ({roi:.2f}%) - agent perd de l\'argent')
    elif roi < 5:
        issues.append(f'ROI faible ({roi:.2f}%) - performance insuffisante')
    
    if issues:
        print('Problèmes détectés:')
        for i, issue in enumerate(issues, 1):
            print(f'  {i}. {issue}')
    else:
        print('✅ Aucun problème critique détecté')
    
    print('\n' + '='*70)
    print('📊 RÉSUMÉ DE L\'AUDIT')
    print('='*70)
    
    summary = {
        'Capital actuel': f'${env.initial_balance:,.0f}',
        'Capital recommandé': '$1,000',
        'ROI observé': f'{roi:.2f}%',
        'Capacité d\'apprentissage': 'À améliorer' if improvement <= 0 else 'Positive',
        'Recommandation': 'Nécessite intégration Memory Decoder'
    }
    
    for key, value in summary.items():
        print(f'{key:.<25} {value}')
    
    print('\n✅ Audit terminé')
    
    return {
        'current_capital': env.initial_balance,
        'roi': roi,
        'learning_improvement': improvement if 'improvement' in locals() else 0,
        'issues': issues
    }

if __name__ == "__main__":
    results = run_complete_audit()
