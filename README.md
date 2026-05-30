# Quantitative Finance

**MSc Medical Physics → Quantitative Research & Energy Markets**

`Python` · `C++` · `pandas` · `numpy` · `scikit-learn`

---

Quantitative researcher with an MSc in Medical Physics transitioning into
systematic trading and energy market modelling. Background in signal processing,
large-scale data pipelines, Monte Carlo simulation, and rigorous experimental
validation — applied here to financial markets.

The mathematical tools that reconstruct an MRI image from noisy k-space data
are the same tools that decompose a financial time series into its underlying
signal components.

---

## Projects

### [`energy_markets/`](./energy_markets/)
**UK Day-Ahead Power Price Forecasting**

End-to-end forecasting pipeline for hourly UK electricity prices using weather,
demand, and calendar features. Implements walk-forward cross-validation to
simulate real day-ahead forecasting conditions.

```bash
cd energy_markets
pip install -r requirements.txt
python3 power_price_forecast.py
```

Results (synthetic 2022–2024 data, walk-forward validation):
- MAE: £4.81/MWh (8.7% of mean price)
- RMSE: £10.23/MWh
- Top predictive feature: 168h lagged price (weekly seasonality)

---

## Coming Soon

- `time_series/` — ARIMA, XGBoost, and LSTM forecasting comparison
- `backtesting/` — Momentum factor research on equity and energy futures
- `cpp_monte_carlo/` — European option pricing and gas storage valuation in C++

---

## Literature

### Energy Markets & Price Forecasting
- **Eydeland & Wolyniec** — *Energy and Power Risk Management* (2003). The standard
  reference for energy market modelling — covers price dynamics, seasonality, and
  spike behaviour that motivated the feature engineering in this repo.
- **Weron, R.** — *Electricity price forecasting: A review of the state-of-the-art*
  (2014), Energy Economics. Comprehensive survey of forecasting methods — justifies
  the use of lagged prices, weather features, and walk-forward validation.
- **Lago et al.** — *Forecasting day-ahead electricity prices: A review of
  state-of-the-art algorithms* (2021), Renewable and Sustainable Energy Reviews.
  Benchmarks gradient boosting and deep learning against classical methods on real
  market data.

### Time-Series Forecasting & Feature Engineering
- **Hyndman & Athanasopoulos** — *Forecasting: Principles and Practice* (free at
  otexts.com/fpp3). Covers ARIMA, seasonality decomposition, and cross-validation
  for time-series — theoretical basis for the walk-forward validation approach.
- **Chen & Guestrin** — *XGBoost: A Scalable Tree Boosting System* (2016), KDD.
  Original paper behind the gradient boosting implementation used in this project.

### Quantitative Finance — General
- **Lopez de Prado, M.** — *Advances in Financial Machine Learning* (2018). Covers
  the specific pitfalls of applying ML to financial data — look-ahead bias,
  overfitting, and correct cross-validation. Directly motivated the walk-forward
  validation approach here rather than random k-fold.
- **Hull, J.** — *Options, Futures, and Other Derivatives* (2022). Standard reference
  for derivatives pricing and energy market instruments — relevant to the Monte Carlo
  project in `cpp_monte_carlo/`.

### Physics → Finance
- **Mantegna & Stanley** — *An Introduction to Econophysics* (1999). Classic text
  connecting statistical physics methods to financial market analysis — bridges the
  gap between MRI physics and quantitative finance thinking.

---

## Contact

[LinkedIn](https://www.linkedin.com/in/lewis-jenkins-565605190/) · [Email](lewis.jenkins96@gmail.com)

*Also see: [mri-physics](https://github.com/ljenkins17/mri-physics) ·
[ml-engineering](https://github.com/ljenkins17/ml-engineering)*