import pandas as pd
import numpy as np
from scipy.optimize import minimize
from benchmark_costs_script import Benchmark
import gym
import torch
from gym import spaces
from stable_baselines3 import SAC
import json
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Step 1: Define the Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, data, total_shares=1000, max_trade_size=100):
        super(TradingEnv, self).__init__()
        self.data = data
        self.total_shares = total_shares
        self.current_step = 0
        self.max_trade_size = max_trade_size

        # Define action space: Continuous, from 0 to max_trade_size
        self.action_space = spaces.Box(low=0, high=self.max_trade_size, shape=(1,), dtype=np.float32)

        # Define observation space (state space)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)

        # Initialize the 'done' flag
        self.done = False

    def reset(self):
        # Reset environment for a new episode
        self.current_step = 0
        self.total_shares = 1000  # Reset the total shares if needed
        self.done = False
        return self._get_state()

    def _get_state(self):
        # Get the current state from the data
        row = self.data.iloc[self.current_step]
        # Use normalized features: top 5 bid and ask prices
        state = np.array([
            row['bid_price_1'], row['bid_price_2'], row['bid_price_3'], row['bid_price_4'], row['bid_price_5'],
            row['ask_price_1'], row['ask_price_2'], row['ask_price_3'], row['ask_price_4'], row['ask_price_5']
        ])
        return state

    def step(self, action):
        # Execute trade of size 'action' (continuous value)
        trade_size = min(action[0], self.total_shares)  # Clip to total shares remaining
        row = self.data.iloc[self.current_step]

        # Simulate transaction cost calculation
        bid_price = row['bid_price_1']
        trade_price = bid_price - (0.01 * np.sqrt(trade_size))  # Simplified market impact
        slippage = (bid_price - trade_price) * trade_size
        market_impact = 0.01 * np.sqrt(trade_size) * trade_size

        # Reward: Negative of total transaction cost
        reward = -(slippage + market_impact)

        # Update state
        self.total_shares -= trade_size
        self.current_step += 1

        # Determine if the episode is done
        if self.current_step >= len(self.data) - 1 or self.total_shares <= 0:
            self.done = True

        return self._get_state(), reward, self.done, {}

# Step 2: Define the Objective Function for Optuna
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    tau = trial.suggest_uniform('tau', 0.01, 0.1)

    # Initialize environment
    env = TradingEnv(aapl_data)

    # Define SAC model
    model = SAC(
        "MlpPolicy", env, verbose=0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau
    )

    # Train the model
    model.learn(total_timesteps=3900)

    # Evaluate model performance (sum of negative rewards as cost)
    obs = env.reset()
    total_cost = 0
    while not env.done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_cost += reward

    # Return negative of total cost to minimize it
    return total_cost

# Step 3: Optimize Hyperparameters Using Optuna
def optimize_hyperparameters(n_trials=50):
    # Create a study to minimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters found: ", study.best_params)
    return study.best_params

# Step 4: Save Trading Schedule as JSON
def generate_trade_schedule(model, data):
    env = TradingEnv(data)
    obs = env.reset()
    trade_schedule = []

    while not env.done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, _, done, _ = env.step(action)

        # Convert action to regular float to ensure JSON serialization compatibility
        trade = {
            "timestamp": str(data.iloc[env.current_step]['timestamp']),
            "shares": float(min(action[0], env.total_shares))  # Convert to float
        }
        trade_schedule.append(trade)

        obs = next_obs

    with open("trade_schedule.json", "w") as f:
        json.dump(trade_schedule, f, indent=4)

    return trade_schedule

# Load the AAPL Dataset
file_path = 'AAPL_Quotes_Data.csv'  # Update with the actual file path if different
aapl_data = pd.read_csv(file_path)
aapl_data['timestamp'] = pd.to_datetime(aapl_data['timestamp'])

# Drop any missing values
aapl_data = aapl_data.dropna()

# Run Optuna to find the best hyperparameters
best_params = optimize_hyperparameters(n_trials=15)

# Train the SAC model using the best hyperparameters
env = TradingEnv(aapl_data)
trained_model = SAC(
    "MlpPolicy", env, verbose=1,
    learning_rate=best_params['learning_rate'],
    batch_size=best_params['batch_size'],
    gamma=best_params['gamma'],
    tau=best_params['tau']
)
trained_model.learn(total_timesteps=3900)
trained_model.save("sac_trade_execution")

# Generate trade schedule JSON
trade_schedule = generate_trade_schedule(trained_model, aapl_data)

def calculate_transaction_cost(trades, data):
    """
    Calculates the transaction cost of the given trades.

    Parameters:
    trades (DataFrame): DataFrame containing trade details including timestamp and shares.
    data (DataFrame): Market data including bid prices.

    Returns:
    float: Total transaction cost.
    """
    total_cost = 0
    for _, trade in trades.iterrows():
        idx = data[data['timestamp'] == trade['timestamp']].index[0]
        bid_price = data.iloc[idx]['bid_price_1']
        trade_price = bid_price - (0.01 * np.sqrt(trade['shares']))  # Simplified market impact
        slippage = (bid_price - trade_price) * trade['shares']
        market_impact = 0.01 * np.sqrt(trade['shares']) * trade['shares']
        total_cost += slippage + market_impact
    return total_cost

def compare_transaction_costs(aapl_data, trained_model, initial_inventory=1000):
    # Step 1: Initialize Benchmark
    benchmark = Benchmark(aapl_data)

    # Step 2: Generate TWAP and VWAP Trades
    twap_trades = benchmark.get_twap_trades(aapl_data, initial_inventory)
    vwap_trades = benchmark.get_vwap_trades(aapl_data, initial_inventory)

    # Step 3: Calculate Transaction Costs for TWAP and VWAP
    twap_cost = calculate_transaction_cost(twap_trades, aapl_data)
    vwap_cost = calculate_transaction_cost(vwap_trades, aapl_data)

    # Step 4: Use the Optimized SAC Model to Generate Trades and Calculate Cost
    env = TradingEnv(aapl_data, total_shares=initial_inventory)
    obs = env.reset()
    sac_trades = []

    while not env.done:
        action, _ = trained_model.predict(obs, deterministic=True)
        next_obs, _, done, _ = env.step(action)
        trade = {
            "timestamp": str(aapl_data.iloc[env.current_step]['timestamp']),
            "shares": min(action[0], env.total_shares)
        }
        sac_trades.append(trade)
        obs = next_obs

    sac_trades_df = pd.DataFrame(sac_trades)
    sac_cost = calculate_transaction_cost(sac_trades_df, aapl_data)

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


# Assuming 'trained_model' is your pre-trained SAC model and 'aapl_data' is your loaded DataFrame
comparison_results = compare_transaction_costs(aapl_data, trained_model, initial_inventory=1000)