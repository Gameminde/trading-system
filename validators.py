"""
Module de validation stricte pour le trading algorithmique
Validation de tous les paramètres d'entrée avec exceptions appropriées
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TradingValidators:
    """Validateurs stricts pour paramètres de trading"""
    
    # Limites de sécurité internes
    _MAX_PRICE_THRESHOLD = 1000000  # $1M max
    _MAX_QUANTITY_THRESHOLD = 1000000  # 1M actions max
    _MAX_CAPITAL_THRESHOLD = 10000000  # $10M max
    _MIN_SYMBOL_LENGTH = 1
    _MAX_SYMBOL_LENGTH = 10
    _VALID_ACTIONS = {'BUY', 'SELL', 'HOLD'}
    
    @staticmethod
    def validate_trade_action(action: str) -> None:
        """Valider action de trading"""
        if not action or not isinstance(action, str):
            raise ValueError(f'Action invalide: {action}. Doit être une chaîne non-vide')
        
        if action.strip() not in TradingValidators._VALID_ACTIONS:
            raise ValueError(f'Action invalide: {action}. Doit être dans {TradingValidators._VALID_ACTIONS}')
    
    @staticmethod
    def validate_quantity(quantity: float) -> None:
        """Valider quantité de trading"""
        if not isinstance(quantity, (int, float)):
            raise ValueError(f'Quantité invalide: {quantity}. Doit être un nombre')
        
        if quantity <= 0:
            raise ValueError(f'Quantité invalide: {quantity}. Doit être positive')
        
        if quantity > TradingValidators._MAX_QUANTITY_THRESHOLD:
            raise ValueError(f'Quantité excessive: {quantity}. Dépasse le seuil de sécurité {TradingValidators._MAX_QUANTITY_THRESHOLD}')
    
    @staticmethod
    def validate_price(price: float) -> None:
        """Valider prix de trading"""
        if not isinstance(price, (int, float)):
            raise ValueError(f'Prix invalide: {price}. Doit être un nombre')
        
        if price <= 0:
            raise ValueError(f'Prix invalide: {price}. Doit être positif')
        
        if price > TradingValidators._MAX_PRICE_THRESHOLD:
            raise ValueError(f'Prix suspect: ${price:,.2f}. Dépasse le seuil de sécurité ${TradingValidators._MAX_PRICE_THRESHOLD:,.2f}')
    
    @staticmethod
    def validate_symbol(symbol: str) -> None:
        """Valider symbole de trading"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f'Symbol invalide: {symbol}. Doit être une chaîne non-vide')
        
        symbol_clean = symbol.strip()
        if len(symbol_clean) < TradingValidators._MIN_SYMBOL_LENGTH:
            raise ValueError(f'Symbole trop court: {symbol}. Longueur minimum: {TradingValidators._MIN_SYMBOL_LENGTH}')
        
        if len(symbol_clean) > TradingValidators._MAX_SYMBOL_LENGTH:
            raise ValueError(f'Symbole trop long: {symbol}. Longueur maximum: {TradingValidators._MAX_SYMBOL_LENGTH}')
        
        # Validation format symbole (lettres + chiffres + points)
        if not re.match(r'^[A-Za-z0-9.]+$', symbol_clean):
            raise ValueError(f'Format symbole invalide: {symbol}. Doit contenir uniquement lettres, chiffres et points')
    
    @staticmethod
    def validate_capital(capital: float) -> None:
        """Valider capital disponible"""
        if not isinstance(capital, (int, float)):
            raise ValueError(f'Capital invalide: {capital}. Doit être un nombre')
        
        if capital <= 0:
            raise ValueError(f'Capital négatif ou nul: {capital}. Doit être positif')
        
        if capital > TradingValidators._MAX_CAPITAL_THRESHOLD:
            raise ValueError(f'Capital excessif: ${capital:,.2f}. Dépasse le seuil de sécurité ${TradingValidators._MAX_CAPITAL_THRESHOLD:,.2f}')
    
    @staticmethod
    def validate_context(context: Dict[str, Any]) -> None:
        """Valider contexte de trading"""
        if not isinstance(context, dict):
            raise ValueError(f'Context invalide: {context}. Doit être un dictionnaire')
        
        if not context:
            raise ValueError('Context vide. Doit contenir au moins une clé')
        
        # Vérifier clés obligatoires
        required_keys = {'timestamp', 'price', 'symbols'}
        missing_keys = required_keys - set(context.keys())
        if missing_keys:
            raise ValueError(f'Context manque clés obligatoires: {missing_keys}')
    
    @staticmethod
    def validate_confidence(confidence: float) -> None:
        """Valider niveau de confiance"""
        if not isinstance(confidence, (int, float)):
            raise ValueError(f'Confiance invalide: {confidence}. Doit être un nombre')
        
        if confidence < 0 or confidence > 1:
            raise ValueError(f'Confiance hors limites: {confidence}. Doit être entre 0 et 1')
    
    @staticmethod
    def validate_risk_score(risk_score: float) -> None:
        """Valider score de risque"""
        if not isinstance(risk_score, (int, float)):
            raise ValueError(f'Score risque invalide: {risk_score}. Doit être un nombre')
        
        if risk_score < 0 or risk_score > 1:
            raise ValueError(f'Score risque hors limites: {risk_score}. Doit être entre 0 et 1')
    
    @staticmethod
    def validate_trade_parameters(action: str, symbol: str, quantity: float, price: float, 
                                context: Dict[str, Any]) -> None:
        """Validation complète des paramètres de trade"""
        try:
            TradingValidators.validate_trade_action(action)
            TradingValidators.validate_symbol(symbol)
            TradingValidators.validate_quantity(quantity)
            TradingValidators.validate_price(price)
            TradingValidators.validate_context(context)
            
            logger.debug(f'✅ Validation réussie pour trade: {action} {quantity} {symbol} @ ${price:.2f}')
            
        except ValueError as e:
            logger.error(f'❌ Validation échouée: {e}')
            raise
    
    @staticmethod
    def validate_portfolio_limits(positions: List[Dict], max_concentration: float = 0.5) -> None:
        """Valider limites du portfolio"""
        if not isinstance(positions, list):
            raise ValueError(f'Positions invalides: {positions}. Doit être une liste')
        
        if max_concentration <= 0 or max_concentration > 1:
            raise ValueError(f'Concentration maximum invalide: {max_concentration}. Doit être entre 0 et 1')
        
        # Calculer concentration par symbole
        symbol_totals = {}
        for pos in positions:
            if not isinstance(pos, dict):
                raise ValueError(f'Position invalide: {pos}. Doit être un dictionnaire')
            
            symbol = pos.get('symbol')
            quantity = pos.get('quantity', 0)
            price = pos.get('entry_price', 0)
            
            if symbol and quantity and price:
                symbol_totals[symbol] = symbol_totals.get(symbol, 0) + (quantity * price)
        
        # Vérifier concentration
        total_value = sum(symbol_totals.values())
        if total_value > 0:
            for symbol, value in symbol_totals.items():
                concentration = value / total_value
                if concentration > max_concentration:
                    raise ValueError(f'Concentration excessive pour {symbol}: {concentration:.1%} > {max_concentration:.1%}')
    
    @staticmethod
    def validate_market_data(market_data) -> None:
        """Valider données de marché"""
        if market_data is None:
            raise ValueError('Données de marché nulles')
        
        if hasattr(market_data, 'empty') and market_data.empty:
            raise ValueError('Données de marché vides')
        
        if hasattr(market_data, 'shape') and market_data.shape[0] == 0:
            raise ValueError('Données de marché sans lignes')
    
    @staticmethod
    def validate_config(config) -> None:
        """Valider configuration de l'agent"""
        if config is None:
            raise ValueError('Configuration nulle')
        
        required_attrs = ['initial_capital', 'max_steps_per_run', 'max_runtime_seconds']
        missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
        
        if missing_attrs:
            raise ValueError(f'Configuration manque attributs: {missing_attrs}')
        
        # Valider valeurs de config
        if hasattr(config, 'initial_capital'):
            TradingValidators.validate_capital(config.initial_capital)
        
        if hasattr(config, 'max_steps_per_run') and config.max_steps_per_run <= 0:
            raise ValueError(f'Max steps invalide: {config.max_steps_per_run}. Doit être positif')
        
        if hasattr(config, 'max_runtime_seconds') and config.max_runtime_seconds <= 0:
            raise ValueError(f'Max runtime invalide: {config.max_runtime_seconds}. Doit être positif')
