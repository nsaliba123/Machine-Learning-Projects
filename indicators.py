import numpy as np
import pandas as pd
from enum import Enum


class Indicators(Enum):
    SMA = 'SMA'
    EMA = 'EMA'
    BBands = 'BBands'
    RSI = 'RSI'
    MACD = 'MACD'
    Stochastic = 'Stochastic'
    ATR = 'ATR'
    GoldenCross = 'GoldenCross'
    DeathCross = 'DeathCross'
    Hurst = 'Hurst'
    Bullish_Fractals = 'Bullish_Fractals'
    Bearish_Fractals = 'Bearish_Fractals'
    Fractal_Strength = 'Fractal_Strength'
    Pct_SMAs = 'Pct_SMAs'
    Pct_EMAs = 'Pct_EMAs'
    SMA_Crossovers = 'SMA_Crossovers'
    SAR = 'SAR'
    Ichimoku = 'Ichimoku'


def sma(data, period):
    data['SMA'] = data['Adj Close'].rolling(window=period).mean()
    return data


def ema(data, period):
    data['EMA'] = data['Adj Close'].ewm(span=period, adjust=False).mean()
    return data


def bbands(data, period):
    sma = data['Adj Close'].rolling(window=period).mean()
    std = data['Adj Close'].rolling(window=period).std()
    data['Upper_BB'] = sma + (std * 2)
    data['Lower_BB'] = sma - (std * 2)
    data['Pct_Upper_BB'] = (
        data['Adj Close'] - data['Upper_BB']) / data['Upper_BB']
    data['Pct_Lower_BB'] = (
        data['Adj Close'] - data['Lower_BB']) / data['Lower_BB']
    return data


def rsi(data, period):
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


def macd(data, period_long, period_short, period_signal):
    ema_long = data['Adj Close'].ewm(span=period_long, adjust=False).mean()
    ema_short = data['Adj Close'].ewm(span=period_short, adjust=False).mean()
    macd = ema_short - ema_long
    data['MACD'] = macd
    data['MACD_Signal'] = macd.ewm(span=period_signal, adjust=False).mean()
    return data


def stochastic_oscillator(data, period):
    low = data['Adj Close'].rolling(window=period).min()
    high = data['Adj Close'].rolling(window=period).max()
    k = 100 * ((data['Adj Close'] - low) / (high - low))
    data['%K'] = k
    data['%D'] = k.rolling(window=period).mean()
    return data


def atr(data, period):
    high_low = data['Adj Close'].diff().abs()
    high_close = (data['Adj Close'] - data['Adj Close'].shift(1)).abs()
    low_close = (data['Adj Close'] - data['Adj Close'].shift(1)).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(window=period).mean()
    return data


def golden_cross(data, period_long, period_short):
    ema_long = data['Adj Close'].ewm(span=period_long, adjust=False).mean()
    ema_short = data['Adj Close'].ewm(span=period_short, adjust=False).mean()
    data['Golden_Cross'] = (ema_short > ema_long) & (
        ema_short.shift(1) < ema_long.shift(1))
    return data


def death_cross(data, period_long, period_short):
    ema_long = data['Adj Close'].ewm(span=period_long, adjust=False).mean()
    ema_short = data['Adj Close'].ewm(span=period_short, adjust=False).mean()
    data['Death_Cross'] = (ema_short < ema_long) & (
        ema_short.shift(1) > ema_long.shift(1))
    return data


def categorize_hurst(hurst):
    if np.isnan(hurst):
        return 0  # Handle potential NaN values in 'hurst'
    elif hurst < 0.48:
        return -1
    elif 0.48 <= hurst < 0.52:
        return 0
    elif hurst >= 0.52:
        return 1


def hurst(data, min_size=100):
    time_series = data['Adj Close'].values
    # Initialize the list with NaN for the first min_size-1 elements
    hurst_values = [np.nan] * (min_size - 1)

    for end_idx in range(min_size, len(time_series) + 1):
        # Extract the subset of the time series up to the current index
        ts_subset = time_series[:end_idx]

        # Calculate Hurst exponent for the current subset
        # Adjust the range of lags based on the current subset size
        lags = range(2, min(end_idx//2, 20))
        tau = [np.std(np.subtract(ts_subset[lag:], ts_subset[:-lag]))
               for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst_exp = reg[0]

        # Append the current Hurst exponent to the list
        hurst_values.append(hurst_exp)

    # Fill the remaining list with NaN if the time series is shorter than expected
    while len(hurst_values) < len(time_series):
        hurst_values.append(np.nan)

    data['Hurst'] = hurst_values
    data['Hurst_Category'] = data['Hurst'].apply(categorize_hurst)

    return data


def bullish_fractals(data):
    bullish_fractal = (
        (data['Low'].shift(4) < data['Low'].shift(5)) &
        (data['Low'].shift(3) < data['Low'].shift(4)) &
        (data['Low'].shift(2) < data['Low'].shift(3)) &
        (data['Low'].shift(1) > data['Low'].shift(2)) &
        (data['Low'] > data['Low'].shift(1))
    )

    data['Bullish_Fractals'] = bullish_fractal
    return data


def bearish_fractals(data):
    bearish_fractal = (
        (data['High'].shift(4) > data['High'].shift(5)) &
        (data['High'].shift(3) > data['High'].shift(4)) &
        (data['High'].shift(2) > data['High'].shift(3)) &
        (data['High'].shift(1) < data['High'].shift(2)) &
        (data['High'] < data['High'].shift(1))
    )

    data['Bearish_Fractals'] = bearish_fractal
    return data


def fractal_strength(data, window):
    bullish_fractal = (
        (data['Low'].shift(4) < data['Low'].shift(5)) &
        (data['Low'].shift(3) < data['Low'].shift(4)) &
        (data['Low'].shift(2) < data['Low'].shift(3)) &
        (data['Low'].shift(1) > data['Low'].shift(2)) &
        (data['Low'] > data['Low'].shift(1))
    )
    bearish_fractal = (
        (data['High'].shift(4) > data['High'].shift(5)) &
        (data['High'].shift(3) > data['High'].shift(4)) &
        (data['High'].shift(2) > data['High'].shift(3)) &
        (data['High'].shift(1) < data['High'].shift(2)) &
        (data['High'] < data['High'].shift(1))
    )

    volatility = data['Adj Close'].pct_change().rolling(window=window).std()
    fractal_strength = ((bullish_fractal | bearish_fractal)
                        * volatility).replace(0, np.nan)
    normalized_strength = (fractal_strength / fractal_strength.max()) * 100

    data['Fractal_Strength'] = normalized_strength
    return data


def percent_diff_smas(data, periods):
    for period in periods:
        data[f'pct_diff_sma_{period}'] = (
            data['Adj Close'] - data['Adj Close'].rolling(window=period).mean()) / data['Adj Close'].rolling(window=period).mean()
    return data


def percent_diff_emas(data, periods):
    for period in periods:
        data[f'pct_diff_ema_{period}'] = (
            data['Adj Close'] - data['Adj Close'].ewm(span=period, adjust=False).mean()) / data['Adj Close'].ewm(span=period, adjust=False).mean()
    return data


def sma_crossovers(data, periods):
    for period in periods:
        period_long = period[0]
        period_short = period[1]
        data[f'SMA_Crossover_{period_long}_{period_short}'] = (data['Adj Close'].rolling(window=period_short).mean(
        ) - data['Adj Close'].rolling(window=period_long).mean()) / data['Adj Close'].rolling(window=period_long).mean()
    return data


def sar(data, acceleration_factor=0.02, max_acceleration_factor=0.2):
    """
    Calculate the Parabolic SAR for a given DataFrame.

    Parameters:
    - data: DataFrame containing 'high', 'low', and 'Adj Close' columns.
    - acceleration_factor: The initial value of the acceleration factor.
    - max_acceleration_factor: The maximum value of the acceleration factor.

    Returns:
    - A DataFrame with the original columns and a new 'parabolic_sar' column.
    """
    # Initialize the first SAR value to the first period's low
    sar = data['Low'][0]
    # Initialize the first EP (extreme point) to the first period's high
    ep = data['High'][0]
    # Initial trend is up
    uptrend = True
    # Initialize acceleration factor
    af = acceleration_factor

    sar_values = [sar]

    for i in range(1, len(data)):
        prev_sar = sar

        # Calculate today's SAR
        sar = sar + af * (ep - sar)

        # Adjust SAR if it's within or beyond today's or yesterday's price range
        if uptrend:
            sar = min(sar, data['Low'][i - 1], data['Low'][i])
            if data['High'][i] > ep:
                ep = data['High'][i]
                af = min(af + acceleration_factor, max_acceleration_factor)
            if data['Low'][i] < sar:
                uptrend = False
                sar = ep
                ep = data['low'][i]
                af = acceleration_factor
        else:
            sar = max(sar, data['High'][i - 1], data['High'][i])
            if data['Low'][i] < ep:
                ep = data['Low'][i]
                af = min(af + acceleration_factor, max_acceleration_factor)
            if data['High'][i] > sar:
                uptrend = True
                sar = ep
                ep = data['High'][i]
                af = acceleration_factor

        sar_values.append(sar)

    data['SAR'] = sar_values
    return data


def ichimoku_cloud(data, period_conversion_line, period_base_line, period_lagging_span, period_displacement):
    conversion_line = (data['High'].rolling(window=period_conversion_line).max(
    ) + data['Low'].rolling(window=period_conversion_line).min()) / 2
    base_line = (data['High'].rolling(window=period_base_line).max(
    ) + data['Low'].rolling(window=period_base_line).min()) / 2
    leading_span_a = ((conversion_line + base_line) /
                      2).shift(period_displacement)
    leading_span_b = ((data['High'].rolling(window=period_lagging_span).max(
    ) + data['Low'].rolling(window=period_lagging_span).min()) / 2).shift(period_displacement)
    data['Conversion_Line'] = conversion_line
    data['Base_Line'] = base_line
    data['Leading_Span_A'] = leading_span_a
    data['Leading_Span_B'] = leading_span_b
    return data


def add_indicators(data, indicators: list[Indicators], verbose=False):
    if verbose:
        print('Calculating indicators...')
    for indicator in indicators:
        if verbose:
            print(f'Calculating {indicator}...')
        if indicator == Indicators.SMA:
            data = sma(data, 20)
        elif indicator == Indicators.EMA:
            data = ema(data, 20)
        elif indicator == Indicators.BBands:
            data = bbands(data, 20)
        elif indicator == Indicators.RSI:
            data = rsi(data, 14)
        elif indicator == Indicators.MACD:
            data = macd(data, 26, 12, 9)
        elif indicator == Indicators.Stochastic:
            data = stochastic_oscillator(data, 14)
        elif indicator == Indicators.ATR:
            data = atr(data, 14)
        elif indicator == Indicators.GoldenCross:
            data = golden_cross(data, 50, 200)
        elif indicator == Indicators.DeathCross:
            data = death_cross(data, 50, 200)
        elif indicator == Indicators.Hurst:
            data = hurst(data)
        elif indicator == Indicators.Bullish_Fractals:
            data = bullish_fractals(data)
        elif indicator == Indicators.Bearish_Fractals:
            data = bearish_fractals(data)
        elif indicator == Indicators.Fractal_Strength:
            data = fractal_strength(data, 6)
        elif indicator == Indicators.Pct_SMAs:
            data = percent_diff_smas(data, [2, 10, 25, 100])
        elif indicator == Indicators.Pct_EMAs:
            data = percent_diff_emas(data, [2, 10, 25, 100])
        elif indicator == Indicators.SMA_Crossovers:
            data = sma_crossovers(data, [(2, 10), (10, 25), (25, 100)])
        elif indicator == Indicators.SAR:
            data = sar(data)
        elif indicator == Indicators.Ichimoku:
            data = ichimoku_cloud(data, 9, 26, 52, 26)

    return data
