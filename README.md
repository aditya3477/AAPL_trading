# AAPL_trading
## Prerequisites
Before running this project, ensure that you have the following installed:

1. Python 3.x
2. All the libraries using the command
 ```console
pip install -r requirements. txt
```

# Setup

First clone the repository. The files and folders are structured in a way, to run the main code.

# Output

The trading schedule is stored in a file called trading_schedule.json (for SAC.py) and trading_schedule_optimized.json (for SAC_optimized.py). The models TWAP, VWAP and the SAC are compared on the transaction costs, for one day (390 minutes).

One of the first things you will notice with the trading schedule json file is that the number of shares traded keep decreasing in descending order. The possible reasons for this are:
1. Gradual Liquidation Strategy: The model is designed to spread out the trading volume to avoid a sudden, large trade that could significantly impact the market price. This is a common strategy in algorithmic trading, known as twap (time-weighted average price).
2. Inventory Management: The model is programmed to decrease the trade size as its inventory shrinks. If the model has fewer shares left, it may reduce the trade size accordingly.
3. Changing Market Conditions: The model adapts to market signals, such as price volatility, volume, and bid-ask spread. If the conditions become less favorable, the model could decide to reduce trade size to minimize exposure to potential adverse price movements.
4. Reward Function Design: The model's behavior is shaped by the reward function. Since the reward function penalizes high transaction costs, slippage, and market impact, the model learns to minimize these costs. A gradual reduction in trade size can be an optimal strategy to avoid incurring high costs, especially when market conditions are unpredictable.

# Models Implemented

The primary model used here is SAC (Soft Actor Critic). Aside from this PPO (Proximal Policy Optimization) was implemented, but this model resulted in a transaction cost around $100 for the given timesteps. 
