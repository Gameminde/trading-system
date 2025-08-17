"""
🧠 MPS WEEK 2 - PORTFOLIO BENCHMARK IMPLEMENTATION
Matrix Product States Portfolio Optimization vs Classical Covariance

RESEARCH REFERENCES:
- 8 Applications Documentées des Tensor Networks en Fi.md, lignes 21-27:
  "L'optimisation de portefeuille via tensor networks transforme radicalement l'approche 
   traditionnelle de Markowitz. La recherche valide ces méthodes sur 8 années de données 
   réelles couvrant 52 actifs avec améliorations significatives des ratios de Sharpe."
   
- 4 Ressources d'Apprentissage Avancées, lignes 72-75:
  "Les Matrix Product States révolutionnent le pricing d'options path-dependent. 
   Pour les options asiatiques, l'approche MPS permet un scaling linéaire avec le nombre 
   de pas temporels, contre un scaling exponentiel pour les arbres binomiaux classiques."

TARGET: ≥10x speedup + ≤1% écart optimal weights vs classical covariance matrix
ETF PORTFOLIO: SPY, QQQ, IWM, TLT (4 assets, 1 year daily data)
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [MPS_BENCHMARK] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Results structure for MPS vs Classical benchmark"""
    
    # Performance Metrics
    mps_time: float = 0.0
    classical_time: float = 0.0
    speedup_factor: float = 0.0
    
    # Memory Usage  
    mps_memory_mb: float = 0.0
    classical_memory_mb: float = 0.0
    memory_reduction: float = 0.0
    
    # Accuracy Comparison
    weight_difference: float = 0.0  # L2 norm difference
    max_weight_error: float = 0.0   # Max absolute error
    relative_error: float = 0.0     # Relative error percentage
    
    # Portfolio Metrics
    mps_weights: List[float] = None
    classical_weights: List[float] = None
    mps_expected_return: float = 0.0
    classical_expected_return: float = 0.0
    
    # Bond Dimension Analysis
    bond_dimensions: List[int] = None
    compression_ratio: float = 0.0
    
    def __post_init__(self):
        if self.mps_weights is None:
            self.mps_weights = []
        if self.classical_weights is None:
            self.classical_weights = []
        if self.bond_dimensions is None:
            self.bond_dimensions = []


class PenroseDiagramGenerator:
    """
    Generate Penrose notation diagrams for MPS portfolio representation
    
    RESEARCH REFERENCE:
    - AGENT_MEMORY.md, lignes 28-31: "TensorNetwork.org: Notation Penrose, diagrammes intuitifs"
    - 4 Ressources d'Apprentissage Avancées, lignes 33-36: "L'approche de Strang privilégie 
      l'intuition géométrique et les applications pratiques"
    """
    
    @staticmethod
    def generate_mps_diagram(n_assets: int, bond_dims: List[int]) -> str:
        """Generate ASCII Penrose diagram for MPS portfolio structure"""
        
        diagram = "🧠 PENROSE NOTATION - MPS PORTFOLIO STRUCTURE\n"
        diagram += "="*60 + "\n"
        
        # Asset labels
        assets = ["SPY", "QQQ", "IWM", "TLT"][:n_assets]
        
        # Top row - asset labels
        diagram += "Assets:  "
        for i, asset in enumerate(assets):
            diagram += f"{asset:>8}"
        diagram += "\n"
        
        # Middle row - MPS structure
        diagram += "MPS:     "
        for i in range(n_assets):
            if i == 0:
                diagram += f"[{bond_dims[0] if bond_dims else 2}]──"
            elif i == n_assets - 1:
                diagram += f"──[{bond_dims[i-1] if i-1 < len(bond_dims) else 2}]"
            else:
                left_bond = bond_dims[i-1] if i-1 < len(bond_dims) else 2
                right_bond = bond_dims[i] if i < len(bond_dims) else 2
                diagram += f"─[{left_bond}]──[{right_bond}]─"
        diagram += "\n"
        
        # Bottom row - bond dimensions  
        diagram += "Bonds:   "
        for i in range(n_assets - 1):
            bond_dim = bond_dims[i] if i < len(bond_dims) else 2
            diagram += f"    {bond_dim:2d}    "
        diagram += "\n"
        
        diagram += "\n💡 Interpretation:"
        diagram += f"\n   • {n_assets} assets connected in MPS chain"
        diagram += f"\n   • Bond dimensions control correlation complexity"
        diagram += f"\n   • Larger bonds = more correlations captured"
        diagram += f"\n   • Trade-off: accuracy vs computational cost"
        
        return diagram
        
    @staticmethod 
    def generate_contraction_diagram() -> str:
        """Generate contraction process diagram"""
        
        diagram = "\n🧩 TENSOR CONTRACTION PROCESS\n"
        diagram += "="*40 + "\n"
        
        diagram += "Classical Covariance (4×4):\n"
        diagram += "┌─────────────────┐\n"
        diagram += "│ SPY QQQ IWM TLT │\n" 
        diagram += "│ ┌───┬───┬───┬───┤\n"
        diagram += "│ │σ² │cov│cov│cov│\n"
        diagram += "│ ├───┼───┼───┼───┤\n"
        diagram += "│ │cov│σ² │cov│cov│\n"
        diagram += "│ ├───┼───┼───┼───┤\n"
        diagram += "│ │cov│cov│σ² │cov│\n"
        diagram += "│ ├───┼───┼───┼───┤\n"
        diagram += "│ │cov│cov│cov│σ² │\n"
        diagram += "└─┴───┴───┴───┴───┘\n"
        
        diagram += "\nMPS Representation:\n"
        diagram += "[A₁]──[A₂]──[A₃]──[A₄]\n"
        diagram += " │     │     │     │  \n"
        diagram += "SPY   QQQ   IWM   TLT\n"
        
        diagram += "\n⚡ Advantage: Compressed correlation structure\n"
        diagram += "📊 Memory: O(n×d²) vs O(n²) classical\n"
        diagram += "🚀 Speed: Linear vs quadratic scaling\n"
        
        return diagram


class MPSPortfolioOptimizer:
    """
    MPS-based Portfolio Optimization Implementation
    
    RESEARCH REFERENCE:
    - 8 Applications Documentées, lignes 44-47: "MPS Imaginary Time Evolution s'avère 
      particulièrement efficace pour l'exploration de l'espace des solutions d'investissement.
      L'intégration de contraintes réalistes maintient la tractabilité computationnelle."
    """
    
    def __init__(self, n_assets: int = 4, max_bond_dim: int = 4):
        self.n_assets = n_assets
        self.max_bond_dim = max_bond_dim
        self.mps_tensors = []
        self.bond_dimensions = []
        logger.info(f"🧠 MPS Portfolio Optimizer initialized: {n_assets} assets, max bond {max_bond_dim}")
        
    def construct_mps_from_returns(self, returns: np.ndarray) -> List[np.ndarray]:
        """
        Construct MPS representation from asset returns data
        
        APPROACH: SVD-based MPS construction with bond dimension optimization
        """
        logger.info("🏗️ Constructing MPS from returns data via SVD decomposition")
        
        # Normalize returns for numerical stability
        returns_normalized = (returns - returns.mean(axis=0)) / returns.std(axis=0)
        
        # Create correlation tensor (simplified 4D tensor for 4 assets)
        correlation_matrix = np.corrcoef(returns_normalized.T)
        
        # MPS construction via iterative SVD
        mps_tensors = []
        bond_dims = []
        
        # Start with correlation matrix as initial tensor
        current_tensor = correlation_matrix
        
        for i in range(self.n_assets - 1):
            # Reshape for SVD
            if i == 0:
                # First tensor: no left bond
                tensor_shape = (1, current_tensor.shape[0], min(self.max_bond_dim, current_tensor.shape[1]))
                U, S, Vt = np.linalg.svd(current_tensor)
                
                # Keep top singular values
                rank = min(self.max_bond_dim, len(S))
                bond_dims.append(rank)
                
                # First MPS tensor  
                mps_tensor = U[:, :rank].reshape(1, current_tensor.shape[0], rank)
                mps_tensors.append(mps_tensor)
                
                # Update for next iteration
                current_tensor = np.diag(S[:rank]) @ Vt[:rank, :]
                
            else:
                # Middle tensors: left bond, physical, right bond
                left_bond = bond_dims[-1]
                
                # SVD decomposition
                U, S, Vt = np.linalg.svd(current_tensor)
                
                rank = min(self.max_bond_dim, len(S))
                if i == self.n_assets - 2:
                    rank = min(rank, current_tensor.shape[1])  # Last bond
                    
                bond_dims.append(rank)
                
                # Construct MPS tensor
                mps_tensor = U[:, :rank].reshape(left_bond, 1, rank)
                mps_tensors.append(mps_tensor)
                
                # Update for next iteration  
                current_tensor = np.diag(S[:rank]) @ Vt[:rank, :]
        
        # Final tensor (no right bond)
        # Adjust shape based on actual tensor dimensions
        if current_tensor.size == bond_dims[-1]:
            final_tensor = current_tensor.reshape(bond_dims[-1], 1)
        else:
            # Take first column if tensor is 2D
            if current_tensor.ndim == 2:
                final_tensor = current_tensor[:, 0].reshape(bond_dims[-1], 1)
            else:
                final_tensor = current_tensor.flatten()[:bond_dims[-1]].reshape(bond_dims[-1], 1)
        mps_tensors.append(final_tensor)
        
        self.mps_tensors = mps_tensors
        self.bond_dimensions = bond_dims
        
        logger.info(f"✅ MPS construction complete: bond dimensions {bond_dims}")
        return mps_tensors
        
    def optimize_portfolio_mps(self, returns: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
        """
        Portfolio optimization using MPS representation
        
        ALGORITHM: Imaginary time evolution for ground state (minimum risk)
        """
        start_time = time.time()
        logger.info("🚀 Starting MPS portfolio optimization")
        
        # Construct MPS from data
        mps_tensors = self.construct_mps_from_returns(returns)
        
        # Expected returns
        expected_returns = returns.mean(axis=0)
        
        # Simplified MPS optimization (proof of concept)
        # In production: would use full imaginary time evolution
        
        # Extract portfolio weights from MPS structure
        weights = np.zeros(self.n_assets)
        
        # Contract MPS tensors to get effective covariance representation
        effective_matrix = self._contract_mps_to_matrix(mps_tensors)
        
        # Mean-variance optimization on compressed representation
        inv_cov = np.linalg.pinv(effective_matrix + 1e-6 * np.eye(self.n_assets))
        
        # Markowitz solution with MPS-compressed covariance
        ones = np.ones(self.n_assets)
        A = ones.T @ inv_cov @ ones
        B = expected_returns.T @ inv_cov @ ones  
        C = expected_returns.T @ inv_cov @ expected_returns
        
        # Optimal weights (mean-variance efficient)
        lambda_param = risk_aversion  # Risk aversion parameter
        weights = inv_cov @ (expected_returns + lambda_param * ones) / (B + lambda_param * A)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        optimization_time = time.time() - start_time
        logger.info(f"✅ MPS optimization complete in {optimization_time:.4f}s")
        
        return weights
        
    def _contract_mps_to_matrix(self, mps_tensors: List[np.ndarray]) -> np.ndarray:
        """Contract MPS tensors to effective covariance matrix"""
        
        # Simplified contraction for proof of concept
        # Full implementation would use proper tensor contraction
        
        n = len(mps_tensors)
        effective_matrix = np.eye(self.n_assets)
        
        # Build effective covariance from MPS structure
        for i in range(n):
            tensor = mps_tensors[i]
            
            # Extract correlation information from tensor structure
            if tensor.ndim == 3:
                # Average over bond dimensions to get effective correlations
                correlation_contribution = np.mean(tensor, axis=(0, 2))
                
                # Update effective matrix (simplified)
                for j in range(min(len(correlation_contribution), self.n_assets)):
                    effective_matrix[i, j] = correlation_contribution[j] if j < len(correlation_contribution) else effective_matrix[i, j]
                    effective_matrix[j, i] = effective_matrix[i, j]  # Symmetry
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(effective_matrix)
        eigenvals = np.maximum(eigenvals, 1e-6)  # Floor eigenvalues
        effective_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return effective_matrix
        
    def get_compression_stats(self) -> Dict[str, float]:
        """Calculate compression statistics"""
        
        if not self.mps_tensors:
            return {"compression_ratio": 0.0}
            
        # Classical storage: n×n covariance matrix
        classical_params = self.n_assets * self.n_assets
        
        # MPS storage: sum of tensor parameters
        mps_params = sum(tensor.size for tensor in self.mps_tensors)
        
        compression_ratio = classical_params / mps_params if mps_params > 0 else 0.0
        
        return {
            "compression_ratio": compression_ratio,
            "classical_params": classical_params,
            "mps_params": mps_params,
            "memory_reduction": (1 - mps_params/classical_params) * 100
        }


class ClassicalPortfolioOptimizer:
    """Classical Markowitz portfolio optimization for benchmark comparison"""
    
    def __init__(self):
        logger.info("📊 Classical Portfolio Optimizer initialized")
        
    def optimize_portfolio_classical(self, returns: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
        """
        Classical mean-variance portfolio optimization
        
        REFERENCE: Standard Markowitz approach with full covariance matrix
        """
        start_time = time.time()
        logger.info("📊 Starting classical portfolio optimization")
        
        # Calculate sample statistics
        expected_returns = returns.mean(axis=0)
        covariance_matrix = np.cov(returns.T)
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
        eigenvals = np.maximum(eigenvals, 1e-6)
        covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Mean-variance optimization
        inv_cov = np.linalg.inv(covariance_matrix)
        ones = np.ones(len(expected_returns))
        
        A = ones.T @ inv_cov @ ones
        B = expected_returns.T @ inv_cov @ ones
        C = expected_returns.T @ inv_cov @ expected_returns
        
        # Optimal weights
        lambda_param = risk_aversion
        weights = inv_cov @ (expected_returns + lambda_param * ones) / (B + lambda_param * A)
        
        # Normalize
        weights = weights / np.sum(weights)
        
        optimization_time = time.time() - start_time
        logger.info(f"✅ Classical optimization complete in {optimization_time:.4f}s")
        
        return weights


class PortfolioBenchmark:
    """
    Main benchmark class comparing MPS vs Classical portfolio optimization
    
    TARGET METRICS (Week 2 Directive):
    - ≥10x speedup MPS vs Classical
    - ≤1% weight difference maximum  
    - Document bond dimension impact
    - Generate Penrose diagrams
    """
    
    def __init__(self, etf_symbols: List[str] = None):
        self.etf_symbols = etf_symbols or ["SPY", "QQQ", "IWM", "TLT"]
        self.mps_optimizer = MPSPortfolioOptimizer(n_assets=len(self.etf_symbols))
        self.classical_optimizer = ClassicalPortfolioOptimizer()
        self.penrose_generator = PenroseDiagramGenerator()
        
        logger.info(f"🎯 Portfolio Benchmark initialized: {self.etf_symbols}")
        
    def load_etf_data(self, data_path: str = "data/etf_prices.csv") -> pd.DataFrame:
        """Load ETF price data (will create synthetic if file doesn't exist)"""
        
        try:
            data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
            logger.info(f"✅ Loaded ETF data from {data_path}: {len(data)} days")
        except FileNotFoundError:
            logger.warning(f"⚠️ Data file not found, generating synthetic data")
            data = self._generate_synthetic_etf_data()
            # Save for future use
            Path(data_path).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(data_path)
            logger.info(f"💾 Saved synthetic data to {data_path}")
            
        return data
        
    def _generate_synthetic_etf_data(self, n_days: int = 252) -> pd.DataFrame:
        """Generate synthetic ETF price data for benchmarking"""
        
        logger.info(f"🔄 Generating synthetic ETF data: {n_days} days")
        
        np.random.seed(42)  # Reproducible results
        
        # Base parameters for each ETF
        etf_params = {
            "SPY": {"initial": 400, "drift": 0.08, "volatility": 0.16},
            "QQQ": {"initial": 350, "drift": 0.12, "volatility": 0.22}, 
            "IWM": {"initial": 180, "drift": 0.06, "volatility": 0.25},
            "TLT": {"initial": 120, "drift": 0.02, "volatility": 0.12}
        }
        
        # Generate correlated returns
        correlation_matrix = np.array([
            [1.00, 0.85, 0.75, -0.25],  # SPY correlations
            [0.85, 1.00, 0.70, -0.30],  # QQQ correlations  
            [0.75, 0.70, 1.00, -0.20],  # IWM correlations
            [-0.25, -0.30, -0.20, 1.00] # TLT correlations (bonds)
        ])
        
        # Generate correlated random returns
        returns = np.random.multivariate_normal(
            mean=[etf_params[etf]["drift"]/252 for etf in self.etf_symbols],
            cov=correlation_matrix * np.outer(
                [etf_params[etf]["volatility"]/np.sqrt(252) for etf in self.etf_symbols],
                [etf_params[etf]["volatility"]/np.sqrt(252) for etf in self.etf_symbols]
            ),
            size=n_days
        )
        
        # Convert to prices
        prices = {}
        for i, etf in enumerate(self.etf_symbols):
            initial_price = etf_params[etf]["initial"]
            price_series = initial_price * np.cumprod(1 + returns[:, i])
            prices[etf] = price_series
            
        # Create DataFrame
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        data = pd.DataFrame(prices, index=dates)
        data.index.name = 'Date'
        
        return data
        
    def run_benchmark(self, risk_aversion: float = 1.0) -> BenchmarkResults:
        """
        Run complete benchmark: MPS vs Classical portfolio optimization
        
        OBJECTIVE: Achieve ≥10x speedup with ≤1% weight difference
        """
        logger.info("🚀 Starting complete MPS vs Classical benchmark")
        
        # Load data
        price_data = self.load_etf_data()
        returns = price_data.pct_change().dropna().values
        
        # Initialize results
        results = BenchmarkResults()
        
        # Generate Penrose diagrams
        penrose_diagram = self.penrose_generator.generate_mps_diagram(
            len(self.etf_symbols), 
            [2, 3, 2]  # Example bond dimensions
        )
        contraction_diagram = self.penrose_generator.generate_contraction_diagram()
        
        print(penrose_diagram)
        print(contraction_diagram)
        
        # Run MPS optimization with timing
        logger.info("⚡ Running MPS optimization...")
        mps_start = time.time()
        mps_weights = self.mps_optimizer.optimize_portfolio_mps(returns, risk_aversion)
        mps_time = time.time() - mps_start
        
        # Run Classical optimization with timing  
        logger.info("📊 Running Classical optimization...")
        classical_start = time.time()
        classical_weights = self.classical_optimizer.optimize_portfolio_classical(returns, risk_aversion)
        classical_time = time.time() - classical_start
        
        # Calculate performance metrics
        speedup = classical_time / mps_time if mps_time > 0 else 0
        weight_diff = np.linalg.norm(mps_weights - classical_weights)
        max_error = np.max(np.abs(mps_weights - classical_weights))
        relative_error = (weight_diff / np.linalg.norm(classical_weights)) * 100
        
        # Get compression stats
        compression_stats = self.mps_optimizer.get_compression_stats()
        
        # Fill results
        results.mps_time = mps_time
        results.classical_time = classical_time 
        results.speedup_factor = speedup
        results.weight_difference = weight_diff
        results.max_weight_error = max_error
        results.relative_error = relative_error
        results.mps_weights = mps_weights.tolist()
        results.classical_weights = classical_weights.tolist()
        results.bond_dimensions = self.mps_optimizer.bond_dimensions.copy()
        results.compression_ratio = compression_stats["compression_ratio"]
        
        # Expected returns
        expected_returns = returns.mean(axis=0)
        results.mps_expected_return = np.dot(mps_weights, expected_returns)
        results.classical_expected_return = np.dot(classical_weights, expected_returns)
        
        # Log results
        logger.info("📊 BENCHMARK RESULTS:")
        logger.info(f"   ⚡ Speedup: {speedup:.2f}x ({'✅' if speedup >= 10 else '⚠️'})")
        logger.info(f"   🎯 Max Weight Error: {max_error:.4f} ({'✅' if max_error <= 0.01 else '⚠️'})")
        logger.info(f"   📈 Compression: {compression_stats['compression_ratio']:.2f}x")
        logger.info(f"   💾 Memory Reduction: {compression_stats.get('memory_reduction', 0):.1f}%")
        
        return results
        
    def save_results(self, results: BenchmarkResults, output_path: str = "logs/MPS_WEEK2_BENCHMARK_RESULTS.json"):
        """Save benchmark results to file"""
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        results_dict = asdict(results)
        results_dict['timestamp'] = time.time()
        results_dict['etf_symbols'] = self.etf_symbols
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        logger.info(f"💾 Results saved to {output_path}")
        
        return output_path


def main():
    """Main execution for Week 2 MPS Portfolio Benchmark"""
    
    logger.info("🧠 MPS WEEK 2 - PORTFOLIO BENCHMARK EXECUTION")
    logger.info("="*60)
    logger.info("TARGET: ≥10x speedup + ≤1% weight difference")
    logger.info("PORTFOLIO: SPY, QQQ, IWM, TLT (4 ETF benchmark)")
    logger.info("="*60)
    
    # Initialize benchmark
    benchmark = PortfolioBenchmark()
    
    try:
        # Run benchmark
        results = benchmark.run_benchmark(risk_aversion=1.0)
        
        # Save results  
        output_path = benchmark.save_results(results)
        
        # Display summary
        print("\n" + "="*60)
        print("🏆 MPS WEEK 2 BENCHMARK COMPLETE")
        print("="*60)
        print(f"⚡ Speedup Factor: {results.speedup_factor:.2f}x")
        print(f"🎯 Weight Accuracy: {results.max_weight_error:.4f} max error")
        print(f"📈 Compression: {results.compression_ratio:.2f}x memory reduction")
        print(f"🔗 Bond Dimensions: {results.bond_dimensions}")
        print(f"💾 Results: {output_path}")
        
        # Success criteria check
        success_speed = results.speedup_factor >= 10.0
        success_accuracy = results.max_weight_error <= 0.01
        
        if success_speed and success_accuracy:
            print("✅ WEEK 2 OBJECTIVES: ALL CRITERIA MET!")
        else:
            print("⚠️ WEEK 2 OBJECTIVES: Partial success - optimization needed")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    results = main()
