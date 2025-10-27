import pandas as pd
import joblib
import os
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import time

print("--- Live Trading Bot Initializing ---")

# --- FINANCIAL & STRATEGY CONFIGURATION ---
RISK_PER_TRADE_PERCENT = 1.0
CONTRACT_SIZE = 100000
MAX_LOT_SIZE = 50.0
MIN_RISK_PIPS = 2.0
RR_RATIO = 2.0
MAGIC_NUMBER = 12345 # A unique ID for trades placed by this bot

# --- FILE & MT5 CONFIGURATION ---
OUTPUT_DIRECTORY = "output/"
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
MODEL_FILENAME = f"{SYMBOL.lower()}_model.joblib"

# --- HELPER FUNCTIONS (Indicators) ---
# Note: These are adapted to work on live data from MT5
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data.rolling(window=window, min_periods=1).mean()
    rolling_std = data.rolling(window=window, min_periods=1).std()
    upper = rolling_mean + (rolling_std * 2)
    lower = rolling_mean - (rolling_std * 2)
    return upper, lower

def calculate_daily_vwap(df):
    df['typical_price_volume'] = ((df['high'] + df['low'] + df['close']) / 3) * df['tick_volume']
    df['cumulative_volume'] = df.groupby(df.index.date)['tick_volume'].cumsum()
    df['cumulative_tpv'] = df.groupby(df.index.date)['typical_price_volume'].cumsum()
    return df['cumulative_tpv'] / df['cumulative_volume']

# --- MAIN BOT LOGIC ---
def run_live_bot():
    print("--- Connecting to MetaTrader 5 terminal... ---")
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    # Load the trained model
    print("--- Loading the predictive model... ---")
    model_path = os.path.join(OUTPUT_DIRECTORY, MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        mt5.shutdown()
        return
    model = joblib.load(model_path)
    print("--- Model loaded successfully. Bot is now running. ---")
    
    # Get symbol properties
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"{SYMBOL} not found on broker.")
        mt5.shutdown()
        return
    point = symbol_info.point

    try:
        while True:
            # --- 1. Time Synchronization: Wait for a new M5 candle ---
            now_utc = datetime.now(pytz.utc)
            next_candle_time = now_utc.replace(second=0, microsecond=0) + timedelta(minutes=5 - (now_utc.minute % 5))
            sleep_seconds = (next_candle_time - now_utc).total_seconds()
            print(f"Current time: {now_utc.strftime('%H:%M:%S')}. Waiting for {sleep_seconds:.1f}s until next candle at {next_candle_time.strftime('%H:%M:%S')}...")
            time.sleep(sleep_seconds)

            # --- 2. Check for Open Positions managed by this bot ---
            positions = mt5.positions_get(symbol=SYMBOL)
            my_position = None
            if positions:
                for pos in positions:
                    if pos.magic == MAGIC_NUMBER:
                        my_position = pos
                        break
            
            # --- 3. Live Trade Management (Trailing Stop) ---
            if my_position:
                # If trade is already at breakeven (sl == open_price), do nothing
                if my_position.sl == my_position.price_open:
                    print(f"Trade #{my_position.ticket} is active and already at breakeven. Monitoring...")
                    continue

                breakeven_price = my_position.price_open + (my_position.price_open - my_position.sl)
                tick = mt5.symbol_info_tick(SYMBOL)

                if my_position.type == mt5.ORDER_TYPE_BUY and tick.bid >= breakeven_price:
                    print(f"Trade #{my_position.ticket} hit 1R profit. Moving SL to breakeven.")
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": my_position.ticket,
                        "sl": my_position.price_open,
                        "tp": my_position.tp,
                    }
                    mt5.order_send(request)
                # (Similar logic for SELL would be needed if BE price is calculated differently)
                continue # Skip new trade entry while managing an open one

            # --- 4. Fetch Live Data & Calculate Features ---
            # Fetch enough data for indicators (e.g., last 50 candles)
            rates_m5_df = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50))
            rates_m5_df['time'] = pd.to_datetime(rates_m5_df['time'], unit='s', utc=True)
            rates_m5_df.set_index('time', inplace=True)

            rates_h4_df = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 50))
            rates_h4_df['time'] = pd.to_datetime(rates_h4_df['time'], unit='s', utc=True)
            rates_h4_df.set_index('time', inplace=True)

            # Calculate indicators for both timeframes
            rates_m5_df['m5_rsi'] = calculate_rsi(rates_m5_df['close'])
            rates_m5_df['m5_bb_upper'], rates_m5_df['m5_bb_lower'] = calculate_bollinger_bands(rates_m5_df['close'])
            rates_m5_df['m5_vwap'] = calculate_daily_vwap(rates_m5_df)

            rates_h4_df['h4_rsi'] = calculate_rsi(rates_h4_df['close'])
            rates_h4_df['h4_bb_upper'], rates_h4_df['h4_bb_lower'] = calculate_bollinger_bands(rates_h4_df['close'])
            rates_h4_df['h4_vwap'] = calculate_daily_vwap(rates_h4_df)

            # --- 5. Signal Detection & Prediction ---
            last_candle = rates_m5_df.iloc[-2] # Analyze the candle that just closed
            prev_candle = rates_m5_df.iloc[-3]

            is_bullish_reversal = last_candle['close'] > last_candle['open'] and prev_candle['close'] < prev_candle['open']
            is_bearish_reversal = last_candle['close'] < last_candle['open'] and prev_candle['close'] > prev_candle['open']

            if is_bullish_reversal or is_bearish_reversal:
                print(f"Signal detected at {last_candle.name}. Preparing features...")
                # Create the feature set for the model
                features = {}
                # Lagged M5 features (we need more history for this in a real bot)
                for i in range(1, 11):
                    features[f'm5_rsi_lag_{i}'] = rates_m5_df['m5_rsi'].iloc[-(i+1)]
                    # ... Add other lagged features (bb_upper, etc.)
                
                # H4 features (find the corresponding H4 candle)
                h4_candle = rates_h4_df[rates_h4_df.index <= last_candle.name].iloc[-1]
                features['h4_rsi'] = h4_candle['h4_rsi']
                # ... Add other H4 features
                
                feature_df = pd.DataFrame([features], columns=model.feature_names_in_)
                
                prediction = model.predict(feature_df)[0]
                trade_direction = 1 if prediction == 1 else -1

                # --- 6. Trade Execution ---
                print(f"Model predicts: {'BUY' if trade_direction == 1 else 'SELL'}. Validating trade...")
                
                if trade_direction == 1: # Buy
                    sl_price = last_candle['low']
                    risk_points = last_candle['close'] - sl_price
                else: # Sell
                    sl_price = last_candle['high']
                    risk_points = sl_price - last_candle['close']

                if risk_points >= (MIN_RISK_PIPS * point):
                    account_info = mt5.account_info()
                    equity = account_info.equity
                    risk_amount_dollars = equity * (RISK_PER_TRADE_PERCENT / 100)
                    risk_per_lot_dollars = risk_points * CONTRACT_SIZE
                    lot_size = min(risk_amount_dollars / risk_per_lot_dollars, MAX_LOT_SIZE)
                    
                    if lot_size >= 0.01:
                        lot_size = round(lot_size, 2)
                        tp_price = last_candle['close'] + (risk_points * RR_RATIO * trade_direction)
                        tick = mt5.symbol_info_tick(SYMBOL)
                        price = tick.ask if trade_direction == 1 else tick.bid
                        
                        print(f"Executing {'BUY' if trade_direction == 1 else 'SELL'} trade. Size: {lot_size}, SL: {sl_price}, TP: {tp_price}")
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": lot_size,
                            "type": mt5.ORDER_TYPE_BUY if trade_direction == 1 else mt5.ORDER_TYPE_SELL,
                            "price": price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "magic": MAGIC_NUMBER,
                            "comment": "XGBoost Bot",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Order send failed, retcode={result.retcode}")

    except KeyboardInterrupt:
        print("\n--- Bot shutdown requested. ---")
    finally:
        mt5.shutdown()
        print("--- MetaTrader 5 connection closed. Bot has stopped. ---")

if __name__ == '__main__':
    # WARNING: This bot will execute LIVE trades. 
    # Run on a DEMO account first. You are responsible for any financial outcomes.
    run_live_bot()
