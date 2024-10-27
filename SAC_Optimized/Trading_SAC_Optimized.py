import gym
import numpy as np
import pandas as pd
import torch
import json
from gym import spaces
from stable_baselines3 import SAC
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from scipy.optimize import minimize
from benchmark_costs_script import Benchmark

class TradingEnvOptimized(gym.Env):
    def __init__(self, data, total_shares=1000, max_trade_size=100, max_steps=390):
        super(TradingEnvOptimized, self).__init__()
        self.data = data
        self.total_shares = total_shares
        self.current_step = 0
        self.max_trade_size = max_trade_size
        self.max_steps = max_steps

        # Define action space: [trade_size, limit_price]
        self.action_space = spaces.Box(low=np.array([1, 0]), high=np.array([self.max_trade_size, 1]), dtype=np.float32)

        # Define observation space (updated state space)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)

        # Initialize the 'done' flag
        self.done = False

    def reset(self):
        self.current_step = 0
        self.total_shares = 1000  # Reset the total shares if needed
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.data):
            self.done = True
            return np.zeros(self.observation_space.shape)  # Placeholder state

        row = self.data.iloc[self.current_step]

        # Optimized state setup based on comprehensive features
        state = np.array([
            row['bid_price_1'], row['ask_price_1'],  # Best bid and ask prices
            row['bid_size_1'], row['ask_size_1'],    # Best bid and ask sizes
            row['ask_price_1'] - row['bid_price_1'], # Spread
            row['volume'],                           # Trading volume
            row['volatility'],                       # Market volatility
            row['log_return'],                       # Recent price change indicator
            row['open'], row['high'], row['low'], row['close']  # OHLC data
        ])
        return state

    def step(self, action):
        trade_size = max(1, min(action[0], self.total_shares))
        row = self.data.iloc[self.current_step]
        bid_price = row['bid_price_1']

        # Calculate limit price using action
        limit_price = action[1] * (row['ask_price_1'] - row['bid_price_1']) + row['bid_price_1']

        # Simulate transaction cost and reward
        if limit_price >= row['ask_price_1']:
            trade_price = limit_price
            slippage = (bid_price - trade_price) * trade_size
            market_impact = 0.01 * np.sqrt(trade_size) * trade_size
            transaction_fee = 0.001 * trade_price * trade_size
        else:
            trade_size = 0
            slippage = 0
            market_impact = 0
            transaction_fee = 0

        reward = -(slippage + market_impact + transaction_fee)

        self.total_shares -= trade_size
        self.current_step += 1

        if self.current_step >= self.max_steps or self.total_shares <= 0:
            self.done = True

        return self._get_state(), reward, self.done, {}

# Step 2: Define the Objective Function for Optuna
def objective_optimized(trial):
    # Suggest hyperparameters for SAC
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    tau = trial.suggest_float('tau', 0.01, 0.1)

    # Initialize the optimized environment
    env = TradingEnvOptimized(aapl_data)

    # Define SAC model
    model = SAC(
        "MlpPolicy", env, verbose=0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau
    )

    # Train the model
    model.learn(total_timesteps=390)

    # Evaluate model performance (sum of negative rewards as cost)
    obs = env.reset()
    total_cost = 0
    while not env.done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_cost += reward

    return total_cost

# Step 3: Optimize Hyperparameters Using Optuna
def optimize_hyperparameters_optimized(n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_optimized, n_trials=n_trials)

    print("Best hyperparameters found: ", study.best_params)
    return study.best_params

import json

# Step 4: Save Trading Schedule as JSON
def generate_trade_schedule_optimized(model, data):
    env = TradingEnvOptimized(data)
    obs = env.reset()
    trade_schedule = []

    while not env.done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, _, done, _ = env.step(action)

        # Convert action to regular float to ensure JSON serialization compatibility
        trade_size = float(min(action[0] * env.total_shares, env.total_shares))

        # Calculate the limit price based on the normalized action
        limit_price = float(action[1] * (data.iloc[env.current_step]['ask_price_1'] - data.iloc[env.current_step]['bid_price_1']) + data.iloc[env.current_step]['bid_price_1'])

        trade = {
            "timestamp": str(data.iloc[env.current_step]['timestamp']),
            "shares": trade_size,
            "limit_price": limit_price
        }
        trade_schedule.append(trade)
        obs = next_obs

    with open("trade_schedule_optimized.json", "w") as f:
        json.dump(trade_schedule, f, indent=4)

    return trade_schedule

aapl_data = pd.read_csv('merged_bid_ask_ohlcv_data.csv')
aapl_data['timestamp'] = pd.to_datetime(aapl_data['timestamp'])

# Drop missing values and handle outliers
aapl_data = aapl_data.dropna()
aapl_data = aapl_data[(aapl_data['volatility'] < aapl_data['volatility'].quantile(0.99)) & (aapl_data['volume'] < aapl_data['volume'].quantile(0.99))]

# Run Optuna to find the best hyperparameters
best_params = optimize_hyperparameters_optimized(n_trials=20)

# Train the SAC model using the best hyperparameters in the new environment
env_optimized = TradingEnvOptimized(aapl_data)
trained_model = SAC(
    "MlpPolicy", env_optimized, verbose=1,
    learning_rate=best_params['learning_rate'],
    batch_size=best_params['batch_size'],
    gamma=best_params['gamma'],
    tau=best_params['tau']
)
trained_model.learn(total_timesteps=390)
trained_model.save("sac_trade_execution_optimized")

# Generate and save trade schedule
trade_schedule = generate_trade_schedule_optimized(trained_model, aapl_data)

def calculate_transaction_cost_optimized(trades, data):
    """
    Calculates the transaction cost of the given trades.

    Parameters:
    trades (DataFrame): DataFrame containing trade details including timestamp, shares, and limit prices.
    data (DataFrame): Market data including bid prices.

    Returns:
    float: Total transaction cost.
    """
    total_cost = 0
    for _, trade in trades.iterrows():
        matching_row = data[data['timestamp'] == trade['timestamp']]
        if matching_row.empty:
            print(f"Warning: No matching data found for timestamp {trade['timestamp']}. Skipping this trade.")
            continue

        idx = matching_row.index[0]
        bid_price = data.iloc[idx]['bid_price_1']

        # Calculate trade price based on the limit price
        trade_price = min(bid_price, trade['limit_price']) if 'limit_price' in trade else bid_price

        # Adjusted transaction cost model
        slippage = (bid_price - trade_price) * trade['shares']
        market_impact = 0.01 * np.sqrt(trade['shares']) * trade['shares']
        total_cost += slippage + market_impact

    return total_cost

def compare_transaction_costs_optimized(aapl_data, trained_model, initial_inventory=1000):
    # Step 1: Initialize Benchmark
    benchmark = Benchmark(aapl_data)

    # Step 2: Generate TWAP and VWAP Trades
    twap_trades = benchmark.get_twap_trades(aapl_data, initial_inventory)
    vwap_trades = benchmark.get_vwap_trades(aapl_data, initial_inventory)

    # Step 3: Calculate Transaction Costs for TWAP and VWAP
    twap_cost = calculate_transaction_cost_optimized(twap_trades, aapl_data)
    vwap_cost = calculate_transaction_cost_optimized(vwap_trades, aapl_data)

    # Step 4: Use the Optimized SAC Model to Generate Trades and Calculate Cost
    env = TradingEnvOptimized(aapl_data, total_shares=initial_inventory, max_steps = 390)
    obs = env.reset()
    sac_trades = []

    while not env.done:
        action, _ = trained_model.predict(obs, deterministic=True)
        next_obs, _, done, _ = env.step(action)

        # Convert action to trade dictionary
        trade_size = float(min(action[0], env.total_shares))
        limit_price = float(action[1] * (aapl_data.iloc[env.current_step]['ask_price_1'] - aapl_data.iloc[env.current_step]['bid_price_1']) + aapl_data.iloc[env.current_step]['bid_price_1'])
        trade = {
            "timestamp": str(aapl_data.iloc[env.current_step]['timestamp']),
            "shares": trade_size,
            "limit_price": limit_price
        }
        sac_trades.append(trade)
        obs = next_obs

    sac_trades_df = pd.DataFrame(sac_trades)
    sac_cost = calculate_transaction_cost_optimized(sac_trades_df, aapl_data)

    # Step 5: Print Comparison Results
    print("Transaction Cost Comparison:")
    print(f"TWAP Transaction Cost: {twap_cost:.2f}")
    print(f"VWAP Transaction Cost: {vwap_cost:.2f}")
    print(f"SAC Transaction Cost: {sac_cost:.2f}")

    # Step 6: Return Results for Further Analysis
    return {
        "TWAP_Cost": twap_cost,
        "VWAP_Cost": vwap_cost,
        "SAC_Cost": sac_cost
    }

comparison_results = compare_transaction_costs_optimized(aapl_data, trained_model, initial_inventory=1000)