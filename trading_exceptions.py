"""
🚨 MODULE D'EXCEPTIONS GRANULAIRES POUR TRADING ALGORITHMIQUE

Fournit une hiérarchie complète d'exceptions avec contexte métier riche,
logging structuré, et suggestions de récupération automatique.

Niveau Production Institutionnel - Diagnostic Expert
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

class TradingError(Exception):
    """🚨 Exception de base pour toutes les erreurs de trading
    
    Fournit contexte métier riche, logging structuré, et suggestions de récupération
    """
    
    def __init__(self, message: str, error_code: str = None, context: Dict = None, 
                 severity: str = 'ERROR', suggested_action: str = None, 
                 recovery_possible: bool = True):
        super().__init__(message)
        self.error_code = error_code or 'GENERIC_ERROR'
        self.context = context or {}
        self.severity = severity  # INFO, WARNING, ERROR, CRITICAL
        self.suggested_action = suggested_action or 'Vérifier logs détaillés'
        self.recovery_possible = recovery_possible
        self.timestamp = datetime.now()
        self.category = self.__class__.__name__.replace('Error', '').lower()
    
    def to_dict(self) -> Dict:
        """Sérialisation complète pour logging JSON"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': str(self),
            'context': self.context,
            'severity': self.severity,
            'suggested_action': self.suggested_action,
            'recovery_possible': self.recovery_possible,
            'category': self.category,
            'timestamp': self.timestamp.isoformat()
        }
    
    def log_structured(self, logger: logging.Logger) -> None:
        """Logging structuré avec contexte complet"""
        log_data = self.to_dict()
        emoji = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'ERROR': '❌', 'CRITICAL': '🚨'}[self.severity]
        
        if self.severity == 'CRITICAL':
            logger.critical(f"{emoji} {self.error_code}: {self} | Action: {self.suggested_action}")
        elif self.severity == 'ERROR':
            logger.error(f"{emoji} {self.error_code}: {self} | Action: {self.suggested_action}")
        elif self.severity == 'WARNING':
            logger.warning(f"{emoji} {self.error_code}: {self} | Action: {self.suggested_action}")
        else:
            logger.info(f"{emoji} {self.error_code}: {self}")


class InsufficientCapitalError(TradingError):
    """💰 Capital insuffisant pour exécuter transaction"""
    
    def __init__(self, required: float, available: float, symbol: str, action: str):
        deficit = required - available
        utilization = (required / available * 100) if available > 0 else float('inf')
        
        message = f'Capital insuffisant pour {action} {symbol}: requis ${required:.2f}, disponible ${available:.2f}'
        
        context = {
            'required_amount': required,
            'available_amount': available, 
            'deficit': deficit,
            'symbol': symbol,
            'action': action,
            'capital_utilization_pct': utilization
        }
        
        if deficit < available * 0.1:  # Déficit < 10% du capital
            suggestion = f'Réduire position de ${deficit:.2f} ou attendre capital additionnel'
            severity = 'WARNING'
        else:
            suggestion = f'Capital insuffisant. Déficit majeur: ${deficit:.2f}'
            severity = 'ERROR'
            
        super().__init__(message, 'INSUFFICIENT_CAPITAL', context, severity, suggestion)


class QuantumOptimizationError(TradingError):
    """⚛️ Erreur lors de l'optimisation quantique"""
    
    def __init__(self, optimization_step: str, positions_count: int = 0, 
                 quantum_score: float = None, underlying_error: str = None):
        message = f'Échec optimisation quantique à l\'étape \'{optimization_step}\' ({positions_count} positions)'
        
        context = {
            'optimization_step': optimization_step,
            'positions_count': positions_count,
            'quantum_score': quantum_score,
            'underlying_error': underlying_error
        }
        
        if positions_count == 0:
            suggestion = 'Ignorer optimisation quantique pour ce cycle (aucune position)'
            severity = 'INFO'
        else:
            suggestion = 'Utiliser optimisation classique en fallback'
            severity = 'WARNING'
            
        super().__init__(message, 'QUANTUM_OPTIMIZATION_FAILED', context, severity, suggestion)


class StaleMarketDataError(TradingError):
    """⏰ Données de marché obsolètes détectées"""
    
    def __init__(self, data_age_seconds: int, max_age_seconds: int, symbol: str = None):
        minutes_old = data_age_seconds // 60
        max_minutes = max_age_seconds // 60
        staleness_ratio = data_age_seconds / max_age_seconds
        
        message = f'Données obsolètes' + (f' pour {symbol}' if symbol else '')
        message += f': {minutes_old}min > limite {max_minutes}min'
        
        context = {
            'data_age_seconds': data_age_seconds,
            'max_age_seconds': max_age_seconds,
            'minutes_old': minutes_old,
            'staleness_ratio': staleness_ratio,
            'symbol': symbol
        }
        
        if staleness_ratio < 2.0:  # Moins de 2x la limite
            suggestion = 'Rafraîchir données immédiatement'
            severity = 'WARNING'
        else:
            suggestion = 'Données critiquement obsolètes - utiliser fallback'
            severity = 'ERROR'
            
        super().__init__(message, 'STALE_MARKET_DATA', context, severity, suggestion)


class ExcessiveRiskError(TradingError):
    """🚨 Niveau de risque critique détecté"""
    
    def __init__(self, risk_score: float, max_allowed: float, risk_factors: List[str], 
                 portfolio_value: float):
        risk_excess = risk_score - max_allowed
        risk_level = 'CRITIQUE' if risk_score > 0.8 else 'ÉLEVÉ' if risk_score > 0.6 else 'MODÉRÉ'
        
        message = f'Risque {risk_level}: {risk_score:.1%} > {max_allowed:.1%}'
        message += f' - Portfolio: ${portfolio_value:,.2f}'
        
        context = {
            'risk_score': risk_score,
            'max_allowed': max_allowed,
            'risk_excess': risk_excess,
            'risk_factors': risk_factors,
            'portfolio_value': portfolio_value,
            'risk_level': risk_level
        }
        
        if risk_score > 0.9:
            suggestion = 'URGENCE: Liquider positions immédiatement'
            severity = 'CRITICAL'
        elif risk_score > 0.8:
            suggestion = 'Réduire positions de 50% immédiatement'
            severity = 'ERROR'
        else:
            suggestion = 'Réduire positions graduellement'
            severity = 'WARNING'
            
        super().__init__(message, 'EXCESSIVE_RISK', context, severity, suggestion)


class DecisionFusionError(TradingError):
    """🧠 Erreur lors de la fusion des décisions"""
    
    def __init__(self, fusion_step: str, contributing_layers: List[str], 
                 layer_confidences: Dict[str, float] = None):
        message = f'Échec fusion décisions à l\'étape \'{fusion_step}\' - Layers: {contributing_layers}'
        
        context = {
            'fusion_step': fusion_step,
            'contributing_layers': contributing_layers,
            'layer_confidences': layer_confidences or {},
            'layers_count': len(contributing_layers)
        }
        
        # Identifier layer avec confiance la plus élevée pour fallback
        if layer_confidences:
            best_layer = max(layer_confidences.items(), key=lambda x: x[1])
            suggestion = f'Utiliser décision du layer {best_layer[0]} (confiance: {best_layer[1]:.1%})'
        else:
            suggestion = 'Utiliser décision HOLD par sécurité'
            
        super().__init__(message, 'DECISION_FUSION_FAILED', context, 'ERROR', suggestion)


class MemoryDecodingError(TradingError):
    """🧠 Erreur lors du décodage de la mémoire"""
    
    def __init__(self, operation: str, memory_size: int, error_details: str, 
                 input_data: Dict = None):
        message = f'Erreur décodage mémoire lors de \'{operation}\' (taille: {memory_size})'
        
        context = {
            'operation': operation,
            'memory_size': memory_size,
            'error_details': error_details,
            'input_data': input_data or {}
        }
        
        if 'context_encoding' in operation:
            suggestion = 'Vérifier format des données de contexte marché'
        elif 'data_validation' in operation:
            suggestion = 'Valider types et formats des données d\'entrée'
        else:
            suggestion = 'Vérifier intégrité de la mémoire et des données'
            
        super().__init__(message, 'MEMORY_DECODING_FAILED', context, 'ERROR', suggestion)


class MemoryCapacityError(TradingError):
    """💾 Erreur de capacité mémoire"""
    
    def __init__(self, current_size: int, max_capacity: int, operation: str):
        utilization = (current_size / max_capacity * 100) if max_capacity > 0 else 0
        
        message = f'Capacité mémoire atteinte: {current_size}/{max_capacity} ({utilization:.1f}%)'
        
        context = {
            'current_size': current_size,
            'max_capacity': max_capacity,
            'utilization_pct': utilization,
            'operation': operation
        }
        
        if utilization > 95:
            suggestion = 'Nettoyer anciennes expériences ou augmenter capacité'
            severity = 'ERROR'
        else:
            suggestion = 'Continuer avec capacité restante'
            severity = 'WARNING'
            
        super().__init__(message, 'MEMORY_CAPACITY_EXCEEDED', context, severity, suggestion)


class RiskAssessmentError(TradingError):
    """🛡️ Erreur lors de l'évaluation des risques"""
    
    def __init__(self, assessment_step: str, portfolio_value: float, error_details: str, 
                 positions_count: int = 0):
        message = f'Échec évaluation risque à l\'étape \'{assessment_step}\' - Portfolio: ${portfolio_value:,.2f}'
        
        context = {
            'assessment_step': assessment_step,
            'portfolio_value': portfolio_value,
            'error_details': error_details,
            'positions_count': positions_count
        }
        
        if 'missing_data' in assessment_step:
            suggestion = 'Vérifier complétude des données de marché et positions'
        elif 'division_by_zero' in assessment_step:
            suggestion = 'Vérifier valeurs de portfolio et positions non-nulles'
        else:
            suggestion = 'Utiliser évaluation risque simplifiée en fallback'
            
        super().__init__(message, 'RISK_ASSESSMENT_FAILED', context, 'ERROR', suggestion)


class TradeExecutionError(TradingError):
    """💰 Erreur lors de l'exécution d'un trade"""
    
    def __init__(self, execution_step: str, action: str, symbol: str, quantity: float, 
                 price: float, failure_reason: str, error_details: str = None):
        message = f'Échec exécution {action} {quantity} {symbol} @ ${price:.2f} - {failure_reason}'
        
        context = {
            'execution_step': execution_step,
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'failure_reason': failure_reason,
            'error_details': error_details,
            'trade_value': quantity * price
        }
        
        if 'validation' in failure_reason:
            suggestion = 'Vérifier paramètres du trade (quantité, prix, symbol)'
        elif 'insufficient_capital' in failure_reason:
            suggestion = 'Vérifier capital disponible et coûts de transaction'
        else:
            suggestion = 'Vérifier état du système et réessayer'
            
        super().__init__(message, 'TRADE_EXECUTION_FAILED', context, 'ERROR', suggestion)


class MarketDataError(TradingError):
    """📊 Erreur liée aux données de marché"""
    
    def __init__(self, data_type: str, symbol: str, error_details: str, 
                 data_source: str = None):
        message = f'Erreur données {data_type} pour {symbol}: {error_details}'
        
        context = {
            'data_type': data_type,
            'symbol': symbol,
            'error_details': error_details,
            'data_source': data_source or 'unknown'
        }
        
        if 'connection' in error_details.lower():
            suggestion = 'Vérifier connectivité réseau et API'
        elif 'format' in error_details.lower():
            suggestion = 'Vérifier format des données reçues'
        else:
            suggestion = 'Utiliser données de fallback ou simulées'
            
        super().__init__(message, 'MARKET_DATA_ERROR', context, 'ERROR', suggestion)


class ConfigurationError(TradingError):
    """⚙️ Erreur de configuration"""
    
    def __init__(self, config_section: str, parameter: str, value: Any, expected_type: str):
        message = f'Configuration invalide: {config_section}.{parameter} = {value} (attendu: {expected_type})'
        
        context = {
            'config_section': config_section,
            'parameter': parameter,
            'current_value': value,
            'expected_type': expected_type
        }
        
        suggestion = f'Corriger {config_section}.{parameter} avec valeur de type {expected_type}'
        super().__init__(message, 'CONFIGURATION_ERROR', context, 'ERROR', suggestion)


# Test des exceptions
if __name__ == "__main__":
    print("🧪 Test des exceptions de trading...")
    
    # Test exception de base
    base_error = TradingError("Test erreur", "TEST_ERROR", {"test": True})
    print(f"✅ Exception de base: {base_error.error_code}")
    
    # Test exception capital insuffisant
    capital_error = InsufficientCapitalError(5000, 3000, "AAPL", "BUY")
    print(f"✅ Capital insuffisant: {capital_error.severity}")
    
    # Test exception risque excessif
    risk_error = ExcessiveRiskError(0.95, 0.8, ["volatility"], 100000)
    print(f"✅ Risque excessif: {risk_error.severity}")
    
    print("✅ Toutes les exceptions testées avec succès!")
