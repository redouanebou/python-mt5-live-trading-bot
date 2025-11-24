# ⚡ Python Live Trading Bot for MetaTrader 5

This is a professional-grade live trading bot written in Python. It runs in a loop and connects directly to the MT5 terminal via the `MetaTrader5` library to execute trades based on a pre-trained machine learning model.

---
 
## Core Features

* **Time Synchronization:** The bot runs on a loop that waits for the *exact* opening of the next M5 candle. This ensures data is always fresh and synchronized.
* **Live Feature Calculation:** Calculates features (RSI, BBands, VWAP) for both the M5 and H4 timeframes in real-time on every new candle.
* **ML Model Integration:** Loads a pre-trained `joblib` model file to make live predictions (Buy, Sell, or Hold) based on the new features.
* **Dynamic Risk Management:**
    * Calculates the Stop Loss based on the last candle's low/high.
    * Calculates the Take Profit based on a fixed 2:1 Risk/Reward ratio.
    * Calculates the **Lot Size** dynamically based on a fixed risk percentage (e.g., 1% of account equity).
* **Trade Execution:** Uses `mt5.order_send` to place the trade directly on the broker's server with the calculated SL, TP, and Lot Size.
* **Live Trade Management:**
    * On every new candle, it checks for open positions managed by its `MAGIC_NUMBER`.
    * It implements a **Trailing Stop to Breakeven**. Once a trade is in 1R (1x Risk) profit, it automatically moves the Stop Loss to the entry price, making the trade risk-free.
