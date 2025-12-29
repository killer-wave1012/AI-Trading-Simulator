#!/usr/bin/env python3
"""
AI Trading Simulator - COMPLETE PROFESSIONAL SYSTEM IN ONE FILE
Educational paper-trading ONLY. No real money, no live trading.

🚀 Features:
- 3 Strategies: MA Crossover, RSI, Momentum
- AI/ML Price Prediction (Decision Tree)
- Realistic backtesting (slippage/commissions)
- Interactive Plotly charts
- Professional metrics (Sharpe, drawdown, win rate)

USAGE: python this_file.py --strategy ma_crossover --use-ai
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 AI Trading Simulator - Educational Paper-Trading System")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data': {
        'path': "data/",
        'filename': "BTCUSD_1h.csv",
        'date_col': "Date",
        'columns': ["Open", "High", "Low", "Close", "Volume"]
    },
    'trading': {
        'initial_balance': 100000.0,
        'commission': 0.001,
        'slippage': 0.0005
    },
    'strategies': {
        'ma_crossover': {'fast_window': 10, 'slow_window': 30},
        'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
        'momentum': {'window': 14, 'threshold': 0.02}
    },
    'ai': {
        'model': "decision_tree",
        'lookback': 20,
        'prediction_threshold': 0.6
    },
    'backtest': {
        'start_date': "2023-06-01",
        'end_date': "2023-12-01"
    }
}

# =============================================================================
# CORE CLASSES
# =============================================================================

@dataclass
class Trade:
    """Single trade record."""
    timestamp: pd.Timestamp
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    value: float
    pnl: float = 0.0

class Portfolio:
    """Virtual trading portfolio with realistic costs."""
    
    def __init__(self, initial_balance: float, commission: float = 0.001, slippage: float = 0.0005):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.position: float = 0.0
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_balance]
    
    def execute_trade(self, timestamp: pd.Timestamp, side: str, price: float, size: float) -> float:
        """Execute trade with commission and slippage."""
        adj_price = price * (1 + self.slippage if side == 'buy' else 1 - self.slippage)
        value = size * adj_price
        commission_cost = value * self.commission
        net_value = value + commission_cost if side == 'sell' else value - commission_cost
        
        if side == 'buy':
            if self.balance < net_value:
                raise ValueError(f"Insufficient balance: {self.balance} < {net_value}")
            self.balance -= net_value
            self.position += size
        else:
            if self.position < size:
                raise ValueError(f"Insufficient position: {self.position} < {size}")
            self.balance += net_value
            self.position -= size
        
        trade = Trade(timestamp=timestamp, side=side, price=adj_price, size=size, value=value)
        self.trades.append(trade)
        
        current_equity = self.balance + (self.position * price)
        self.equity_curve.append(current_equity)
        return current_equity
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate key performance metrics."""
        if not self.trades:
            return {}
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        win_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(win_trades) / len(self.trades)
        drawdown = (equity_series / equity_series.cummax() - 1).min()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': drawdown,
            'sharpe_ratio': sharpe,
            'final_balance': equity_series.iloc[-1],
            'total_trades': len(self.trades)
        }

class DataEngine:
    """Handles loading and preprocessing of market data."""
    
    def __init__(self, config: dict = CONFIG):
        self.config = config['data']
        self.df: Optional[pd.DataFrame] = None
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load OHLC data from CSV file."""
        filepath = Path(self.config['path']) / filename
        
        if not filepath.exists():
            self._generate_sample_data(filename)
            filepath = Path(self.config['path']) / filename
        
        self.df = pd.read_csv(
            filepath,
            parse_dates=[self.config['date_col']],
            index_col=self.config['date_col']
        )
        
        required_cols = self.config['columns']
        for col in required_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        logger.info(f"Loaded {len(self.df)} rows from {filename}")
        return self.df
    
    def get_data_slice(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return data slice for backtesting."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        return self.df[mask].copy()
    
    def _generate_sample_data(self, filename: str):
        """Generate realistic sample BTC data."""
        print(f"📊 Generating sample data: {filename}")
        Path(self.config['path']).mkdir(exist_ok=True)
        
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='1H')
        n = len(dates)
        
        price = 25000
        prices, volumes, highs, lows = [], [], [], []
        
        for i in range(n):
            ret = np.random.normal(0.0002, 0.02)
            price *= (1 + ret)
            prices.append(price)
            highs.append(price * (1 + abs(np.random.normal(0, 0.01))))
            lows.append(price * (1 - abs(np.random.normal(0, 0.01))))
            volume = np.random.normal(1000, 200) * (1 + abs(ret) * 10)
            volumes.append(max(0, volume))
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        })
        
        df.to_csv(Path(self.config['path']) / filename, index=False)
        print(f"✅ Sample data saved to {self.config['path']}{filename}")

# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def get_signal(self, current_bar: pd.Series, timestamp: pd.Timestamp, 
                   historical_data: pd.DataFrame) -> float:
        pass

class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover strategy."""
    
    def get_signal(self, current_bar: pd.Series, timestamp: pd.Timestamp,
                   historical_data: pd.DataFrame) -> float:
        if len(historical_data) < self.config['slow_window']:
            return 0.0
        
        data = historical_data.tail(100).copy()
        data['fast_ma'] = data['Close'].rolling(window=self.config['fast_window']).mean()
        data['slow_ma'] = data['Close'].rolling(window=self.config['slow_window']).mean()
        
        if len(data) < 2:
            return 0.0
        
        current_fast = data['fast_ma'].iloc[-1]
        current_slow = data['slow_ma'].iloc[-1]
        prev_fast = data['fast_ma'].iloc[-2]
        prev_slow = data['slow_ma'].iloc[-2]
        
        if current_fast > current_slow and prev_fast <= prev_slow:
            return 1.0
        elif current_fast < current_slow and prev_fast >= prev_slow:
            return -1.0
        
        ma_signal = (current_fast - current_slow) / current_slow
        return np.clip(ma_signal * 2, -1.0, 1.0)

class RSIStrategy(BaseStrategy):
    """RSI Mean Reversion strategy."""
    
    def get_signal(self, current_bar: pd.Series, timestamp: pd.Timestamp,
                   historical_data: pd.DataFrame) -> float:
        if len(historical_data) < self.config['period']:
            return 0.0
        
        delta = historical_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < self.config['oversold']:
            return 1.0
        elif current_rsi > self.config['overbought']:
            return -1.0
        
        rsi_signal = (50 - current_rsi) / 50
        return rsi_signal * 0.5

class MomentumStrategy(BaseStrategy):
    """Momentum Breakout strategy."""
    
    def get_signal(self, current_bar: pd.Series, timestamp: pd.Timestamp,
                   historical_data: pd.DataFrame) -> float:
        if len(historical_data) < self.config['window']:
            return 0.0
        
        data = historical_data.tail(self.config['window'] * 2).copy()
        momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-self.config['window']] - 1)
        avg_volume = data['Volume'].tail(self.config['window']).mean()
        volume_spike = data['Volume'].iloc[-1] > avg_volume * 1.5
        
        raw_signal = momentum / self.config['threshold']
        signal = raw_signal if volume_spike else raw_signal * 0.5
        return np.clip(signal, -1.0, 1.0)

# =============================================================================
# AI COMPONENT
# =============================================================================

class AIPredictor:
    """ML model to predict next price direction."""
    
    def __init__(self, model_type: str = "decision_tree", lookback: int = 20):
        self.model_type = model_type
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _create_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create ML features from OHLCV data."""
        df = data.copy()
        
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['price_sma_ratio'] = df['Close'] / df['sma_20']
        df['volume_sma'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        for lag in [1, 2, 3]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        feature_cols = [col for col in df.columns if col not in ['target']]
        df_features = df[feature_cols].dropna()
        target = df['target'].loc[df_features.index]
        
        return df_features, target
    
    def train(self, data: pd.DataFrame):
        """Train model on historical data."""
        features, target = self._create_features(data)
        
        if len(features) < 50:
            print("⚠️ Insufficient data for AI training")
            return
        
        if self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            self.model = LogisticRegression(random_state=42)
        
        if self.model_type != "decision_tree":
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled, target)
        else:
            self.model.fit(features, target)
        
        self.is_trained = True
        print(f"✅ AI Model trained on {len(features)} samples")
    
    def predict_direction(self, current_bar: pd.Series, historical_data: pd.DataFrame) -> float:
        """Predict price direction signal (-1 to 1)."""
        if not self.is_trained or len(historical_data) < self.lookback:
            return 0.0
        
        data_slice = historical_data.tail(self.lookback * 2)
        full_data = pd.concat([pd.DataFrame([current_bar]), data_slice]).tail(len(data_slice) + 1)
        features, _ = self._create_features(full_data)
        
        if len(features) == 0:
            return 0.0
        
        latest_features = features.iloc[-1:].dropna(axis=1)
        if latest_features.empty:
            return 0.0
        
        if self.model_type == "decision_tree":
            prob = self.model.predict_proba(latest_features)[0, 1]
        else:
            latest_features_scaled = self.scaler.transform(latest_features)
            prob = self.model.predict_proba(latest_features_scaled)[0, 1]
        
        signal = (prob - 0.5) * 2
        return signal

# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Main backtesting engine."""
    
    def __init__(self, initial_balance: float = 100000.0, commission: float = 0.001):
        self.portfolio = Portfolio(initial_balance, commission)
        self.trade_log: List[Dict] = []
    
    def run(self, data: pd.DataFrame, strategy: BaseStrategy, ai_predictor=None) -> Dict[str, Any]:
        """Run complete backtest."""
        logger.info("Starting backtest...")
        
        for timestamp, row in data.iterrows():
            signal = strategy.get_signal(row, timestamp, data.loc[:timestamp])
            ai_signal = ai_predictor.predict_direction(row, data.loc[:timestamp]) if ai_predictor else 0
            self._execute_signals(timestamp, row['Close'], signal, ai_signal)
        
        metrics = self.portfolio.get_metrics()
        results = {
            'metrics': metrics,
            'trade_log': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve
        }
        
        logger.info(f"Backtest complete. Total return: {metrics['total_return']:.2%}")
        return results
    
    def _execute_signals(self, timestamp: pd.Timestamp, price: float, 
                        strategy_signal: float, ai_signal: float):
        """Execute trades based on combined signals."""
        combined_signal = strategy_signal + ai_signal
        
        if abs(combined_signal) > 1.0:
            side = 'buy' if combined_signal > 0 else 'sell'
            size = self._calculate_position_size(price)
            
            try:
                equity = self.portfolio.execute_trade(timestamp, side, price, size)
                self.trade_log.append({
                    'timestamp': timestamp, 'side': side, 'price': price, 
                    'size': size, 'equity': equity
                })
            except ValueError:
                pass  # Skip invalid trades
    
    def _calculate_position_size(self, price: float) -> float:
        """Position sizing (max 10% of equity)."""
        equity = self.portfolio.balance + (self.portfolio.position * price)
        max_position_value = equity * 0.1
        return max_position_value / price

# =============================================================================
# VISUALIZATION
# =============================================================================

class TradingVisualizer:
    """Creates professional trading charts."""
    
    @staticmethod
    def plot_backtest(data: pd.DataFrame, trades: List[Trade], equity_curve: List[float],
                     metrics: Dict[str, float], save_path: str = "backtest_results.html"):
        """Create comprehensive backtest visualization."""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price Chart with Trades', 'Equity Curve', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                          low=data['Low'], close=data['Close'], name="Price"),
            row=1, col=1
        )
        
        # Trade markers
        buy_trades = [t for t in trades if t.side == 'buy']
        sell_trades = [t for t in trades if t.side == 'sell']
        
        if buy_trades:
            fig.add_trace(go.Scatter(
                x=[t.timestamp for t in buy_trades], y=[t.price for t in buy_trades],
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy', hovertemplate='<b>BUY</b><br>%{x}<br>$%{y:.2f}<extra></extra>'
            ), row=1, col=1)
        
        if sell_trades:
            fig.add_trace(go.Scatter(
                x=[t.timestamp for t in sell_trades], y=[t.price for t in sell_trades],
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell', hovertemplate='<b>SELL</b><br>%{x}<br>$%{y:.2f}<extra></extra>'
            ), row=1, col=1)
        
        # Equity curve
        equity_df = pd.DataFrame({'equity': equity_curve}, index=data.index[:len(equity_curve)])
        fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['equity'], 
                                line=dict(color='blue', width=2), name='Equity'), row=2, col=1)
        
        # Drawdown
        equity_series = pd.Series(equity_curve)
        drawdown = (equity_series / equity_series.cummax() - 1) * 100
        drawdown_df = pd.DataFrame({'drawdown': drawdown}, index=data.index[:len(drawdown)])
        fig.add_trace(go.Scatter(x=drawdown_df.index, y=drawdown_df['drawdown'],
                                line=dict(color='red'), fill='tonexty', name='Drawdown %'), row=3, col=1)
        
        fig.update_layout(
            title=f"AI Trading Backtest Results<br><sup>"
                  f"Return: {metrics['total_return']:.2%} | "
                  f"Win: {metrics['win_rate']:.1%} | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                  f"Trades: {metrics['total_trades']}</sup>",
            height=900, showlegend=True, template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1)
        fig.update_yaxes(title_text="Equity ($)", row=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=3)
        
        fig.write_html(save_path)
        print(f"📊 Interactive charts saved: {save_path}")
        fig.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def create_strategy(strategy_name: str, config: dict) -> BaseStrategy:
    """Strategy factory."""
    strategies = {
        'ma_crossover': MACrossoverStrategy,
        'rsi': RSIStrategy,
        'momentum': MomentumStrategy
    }
    strategy_class = strategies.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategy_class(config['strategies'][strategy_name])

def main():
    parser = argparse.ArgumentParser(description="AI Trading Simulator")
    parser.add_argument('--strategy', '-s', default='ma_crossover',
                       choices=['ma_crossover', 'rsi', 'momentum'],
                       help='Strategy to test')
    parser.add_argument('--data', default='BTCUSD_1h.csv', help='Data file')
    parser.add_argument('--use-ai', action='store_true', help='Enable AI')
    parser.add_argument('--output', '-o', default='backtest_results.html', help='Output file')
    args = parser.parse_args()
    
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Data: {args.data}")
    print(f"AI: {'✅ Enabled' if args.use_ai else '❌ Disabled'}")
    print("-" * 70)
    
    try:
        # 1. Load data
        engine = DataEngine(CONFIG)
        data = engine.load_data(args.data)
        data_slice = engine.get_data_slice(
            CONFIG['backtest']['start_date'],
            CONFIG['backtest']['end_date']
        )
        
        # 2. Setup strategy & AI
        strategy = create_strategy(args.strategy, CONFIG)
        ai_predictor = None
        if args.use_ai:
            ai_predictor = AIPredictor(CONFIG['ai']['model'], CONFIG['ai']['lookback'])
            ai_predictor.train(data_slice)
        
        # 3. Run backtest
        backtester = Backtester(**CONFIG['trading'])
        results = backtester.run(data_slice, strategy, ai_predictor)
        
        # 4. Visualize & report
        TradingVisualizer().plot_backtest(
            data_slice, results['trade_log'], results['equity_curve'],
            results['metrics'], args.output
        )
        
        # 5. Print summary
        metrics = results['metrics']
        print("\n" + "="*70)
        print("📈 BACKTEST RESULTS")
        print("="*70)
        print(f"💰 Final Balance:  ${metrics['final_balance']:,.2f}")
        print(f"📊 Total Return:   {metrics['total_return']:.2%}")
        print(f"🎯 Win Rate:       {metrics['win_rate']:.1%}")
        print(f"📉 Max Drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"⚡ Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
        print(f"🔄 Total Trades:   {metrics['total_trades']}")
        print("="*70)
        print("⚠️  EDUCATIONAL ONLY - NO REAL MONEY TRADING")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main()