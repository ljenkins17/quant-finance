import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def generate_synthetic_power_data(
    start: str = "2022-01-01",
    end: str = "2024-01-01",
    freq: str = "h",
) -> pd.DataFrame:
    np.random.seed(42)
    idx = pd.date_range(start, end, freq=freq)
    n = len(idx)
    t = np.arange(n)

    # Temperature: seasonal cycle + daily cycle + noise (degrees C)
    seasonal_temp = 10 - 8 * np.cos(2 * np.pi * t / (24 * 365.25))
    daily_temp = 3 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 3)
    temperature = seasonal_temp + daily_temp + np.random.randn(n) * 2.0

    # Wind speed: lognormal with weak seasonality (m/s)
    wind_seasonal = 1 + 0.3 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)
    wind_speed = np.random.lognormal(mean=wind_seasonal, sigma=0.5)
    wind_speed = np.clip(wind_speed, 0, 25)

    # Solar irradiance: seasonal + diurnal (W/m2), zero at night
    solar_seasonal = np.clip(300 + 250 * np.sin(2 * np.pi * t / (24 * 365.25)), 0, None)
    hour_of_day = t % 24
    solar_diurnal = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12), 0, 1)
    solar = solar_seasonal * solar_diurnal + np.random.randn(n) * 20
    solar = np.clip(solar, 0, None)

    # Electricity demand (MW): driven by temperature, time of day, weekday/weekend
    demand_base = 35_000
    demand_seasonal = 8_000 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)
    demand_diurnal = 5_000 * (
        0.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 12)
        + 0.3 * np.sin(2 * np.pi * (hour_of_day - 19) / 8)
    )
    # Cold weather increases heating demand
    demand_temp = -300 * (temperature - 10)
    # Weekend demand is lower than weekday
    is_weekend = pd.Series(idx).dt.dayofweek.isin([5, 6]).values.astype(float)
    demand_weekend = -4_000 * is_weekend
    demand = (
        demand_base + demand_seasonal + demand_diurnal
        + demand_temp + demand_weekend
        + np.random.randn(n) * 1_000
    )

    # Power price (£/MWh): driven by demand, wind, solar, and seasonality
    price_base = 60
    price_demand = 0.002 * (demand - demand_base)   # high demand pushes price up
    price_wind = -0.8 * wind_speed                   # wind suppresses price (zero marginal cost)
    price_solar = -0.03 * solar                      # solar suppresses price (zero marginal cost)
    price_seasonal = 15 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)
    # Occasional price spikes from plant outages or demand surges
    price_spike = np.zeros(n)
    spike_idx = np.random.choice(n, size=int(n * 0.005), replace=False)
    price_spike[spike_idx] = np.random.exponential(80, size=len(spike_idx))
    price = (
        price_base + price_demand + price_wind + price_solar
        + price_seasonal + price_spike
        + np.random.randn(n) * 5
    )
    # Negative prices occur in real markets when renewables flood the grid
    price = np.clip(price, -50, 500)

    df = pd.DataFrame({
        "price":       price,
        "demand":      demand,
        "temperature": temperature,
        "wind_speed":  wind_speed,
        "solar":       solar,
    }, index=idx)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Calendar features — capture time-of-day, day-of-week, and seasonal patterns
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)

    # Lagged price features — use 24h, 48h, and 168h (1 week) lags
    # All lags are at least 24h to avoid look-ahead bias in day-ahead forecasting
    for lag in [24, 48, 168]:
        df[f"price_lag_{lag}h"] = df["price"].shift(lag)

    # Rolling statistics on lagged prices — capture recent trend and volatility
    df["price_rolling_mean_24h"] = df["price"].shift(24).rolling(24).mean()
    df["price_rolling_std_24h"]  = df["price"].shift(24).rolling(24).std()
    df["price_rolling_mean_7d"]  = df["price"].shift(24).rolling(24 * 7).mean()

    # Lagged weather and demand features (day-ahead forecast would be available)
    for lag in [24]:
        df[f"demand_lag_{lag}h"]      = df["demand"].shift(lag)
        df[f"temperature_lag_{lag}h"] = df["temperature"].shift(lag)
        df[f"wind_lag_{lag}h"]        = df["wind_speed"].shift(lag)
        df[f"solar_lag_{lag}h"]       = df["solar"].shift(lag)

    # Interaction feature: cold + low wind = high price risk
    df["cold_calm"] = (
        np.clip(10 - df["temperature_lag_24h"], 0, None)
        * np.clip(5 - df["wind_lag_24h"], 0, None)
    )

    # Drop rows with NaN values created by lagging and rolling
    return df.dropna()


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "price",
    train_months: int = 12,
    test_months: int = 1,
) -> pd.DataFrame:
    # Walk-forward validation simulates real forecasting conditions
    # We always train on the past and predict the future — never shuffle time-series data
    results = []
    start = df.index[0]
    end = df.index[-1]
    train_end = start + pd.DateOffset(months=train_months)

    while train_end + pd.DateOffset(months=test_months) <= end:
        test_end = train_end + pd.DateOffset(months=test_months)

        # Split into train and test sets based on time
        train = df[df.index < train_end]
        test  = df[(df.index >= train_end) & (df.index < test_end)]

        if len(test) == 0:
            break

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test  = test[feature_cols].values
        y_test  = test[target_col].values

        # Gradient boosting model — robust to outliers and captures non-linear relationships
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append(pd.DataFrame({
            "actual":    y_test,
            "predicted": y_pred,
        }, index=test.index))

        # Expand the training window by one month
        train_end = test_end

    return pd.concat(results)


def get_feature_importance(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "price",
) -> pd.Series:
    # Train on full dataset to get stable feature importance estimates
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(df[feature_cols].values, df[target_col].values)
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


def plot_results(predictions: pd.DataFrame, importances: pd.Series):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("UK Day-Ahead Power Price Forecasting", fontsize=14, fontweight="bold")

    # Plot first month of predictions vs actuals
    sample = predictions.iloc[:24 * 30]
    axes[0].plot(sample.index, sample["actual"],    label="Actual",    linewidth=1.0, alpha=0.8)
    axes[0].plot(sample.index, sample["predicted"], label="Predicted", linewidth=1.0, alpha=0.8, linestyle="--")
    axes[0].set_ylabel("Price (£/MWh)")
    axes[0].set_title("Actual vs Predicted — First Month of Test Period")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Scatter plot — points clustering along the diagonal indicate good predictions
    axes[1].scatter(predictions["actual"], predictions["predicted"], alpha=0.1, s=5)
    lims = [predictions[["actual", "predicted"]].min().min(),
            predictions[["actual", "predicted"]].max().max()]
    axes[1].plot(lims, lims, "r--", linewidth=1, label="Perfect forecast")
    axes[1].set_xlabel("Actual Price (£/MWh)")
    axes[1].set_ylabel("Predicted Price (£/MWh)")
    axes[1].set_title("Actual vs Predicted — Scatter")
    axes[1].legend()

    # Feature importance — shows which inputs the model relies on most
    importances.head(12).plot(kind="barh", ax=axes[2], color="steelblue")
    axes[2].set_xlabel("Feature Importance")
    axes[2].set_title("Top 12 Predictive Features")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig("power_price_forecast_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved to power_price_forecast_results.png")


def main():
    print("UK Day-Ahead Power Price Forecasting")
    print("=" * 42)

    print("Generating synthetic market data (2022-2024)...")
    df_raw = generate_synthetic_power_data()
    print(f"  {len(df_raw):,} hourly observations")

    print("Engineering features...")
    df = engineer_features(df_raw)
    print(f"  {len(df):,} observations after removing NaNs")

    feature_cols = [c for c in df.columns if c != "price"]
    print(f"  {len(feature_cols)} features")

    print("\nRunning walk-forward validation (12-month train, 1-month test)...")
    predictions = walk_forward_validation(df, feature_cols)

    mae  = mean_absolute_error(predictions["actual"], predictions["predicted"])
    rmse = root_mean_squared_error(predictions["actual"], predictions["predicted"])
    mean_price = predictions["actual"].mean()

    print(f"\n  Test period: {predictions.index[0].date()} to {predictions.index[-1].date()}")
    print(f"  MAE:         £{mae:.2f}/MWh  ({100*mae/mean_price:.1f}% of mean price)")
    print(f"  RMSE:        £{rmse:.2f}/MWh")
    print(f"  Mean price:  £{mean_price:.2f}/MWh")

    print("\nComputing feature importances...")
    importances = get_feature_importance(df, feature_cols)
    print(f"  Top feature: {importances.index[0]} ({importances.iloc[0]:.3f})")

    plot_results(predictions, importances)


if __name__ == "__main__":
    main()