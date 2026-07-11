# Temperature Forecasting with a Feed-Forward Neural Network

This project was developed as **final project** of the course *Machine learning for the Earth System* at the University to Cologne in th summer term 2026. It implements a Feed-Forward Neural Network (FFN) in PyTorch to forecast air temperature one hour ahead using meteorological time series data.

---

## Project Overview

The model is trained on two input variables — air temperature and solar irradiation — measured at the weather station of the Institute of Metorology and Geohysics of the UNiversity to Cologen at the UniSport Cologne. The evaluation focuses exclusively on temperature prediction accuracy.

**Training data:** March, April, May 2024 and March, April, May 2025  
**Test data:** March, April, May 2026  
**Forecast horizon:** One step ahead (1 hour)  
**Input window:** 24 hours  

---

## Repository Structure
ML-project/
├── preprocessing.py       # Load raw .dat files, resample to hourly, normalize
├── model.py               # FFN model definition (PyTorch)
├── train.py               # Training loop
├── evaluate.py            # Evaluation, plots, ablation study
├── data_hourly.csv    # Preprocessed hourly data (temperature + radiation)
├── scale.pkl          # Fitted MinMaxScaler for inverse transformation
├── requirements.txt
└── README.md


> **Note:** Raw data files (`data/`) are not included in this repository due to file size. See **Data** section below.

---

## Data

The data was measured at the weather station at the UniSport Cologne, in 1-minute intervals and downloaded in a 10-minute mean values.

The preprocessing extract the temperature data and the incoming solar irradiation measurements and calculated the houly mean values of it. 

| Variable | Column | Unit |
|---|---|---|
| Air temperature (2m) | `AirTC_2_Avg` | °C |
| Shortwave radiation (upward) | `SWUpper_Avg` | W/m² |

Raw files follow the naming convention `YYYYMMDD_msgg10m.dat` and are  organized in the folder structure `data/MM/DD/`.

The data is visible on the DataBrowser of the weather station: https://atmos.meteo.uni-koeln.de/~citystation/dataBrowser/dataBrowser1.html?site=CITYSTATION&date=0&UpperLeft=Temperatur_K%C3%B6ln and can be requested under wetter-station@uni-koeln.de.



## Model

A simple Feed-Forward Neural Network (FFN) implemented in PyTorch:
Input:  24 timesteps × 2 variables = 48 values
→ Linear(48 → 64) + ReLU
→ Linear(64 → 32) + ReLU
→ Linear(32 → 1)
Output: predicted temperature at t+1

Total parameters: **5,249**

---

## Training

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr = 0.001)
- **Batch size:** 32
- **Epochs:** 50
- **Seed:** 23 (for reproducibility)

---

## Results

Evaluation on test data (March–May 2026):

| Metric | Value |
|---|---|
| MAE | 0.46 °C |
| RMSE | 0.62 °C |

### Day vs. Night Accuracy

The model was evaluated separately for daytime (07:00–20:00)  
and nighttime (20:00–07:00) to assess performance under  
different radiation conditions.

### Ablation Study

To assess the contribution of solar radiation as an input variable,  
an ablation test was conducted: the radiation values in the test set  
were replaced with random Gaussian noise (same mean and std as the  
original data). The resulting increase in MAE and RMSE quantifies  
the importance of radiation for temperature forecasting.

### Monthly Accuracy Distribution

For each month, the distribution of daily prediction accuracy  
(percentage of hours with error < 0.5°C) is visualized as a histogram  
with 10% bins and 25th, 50th, and 75th percentile markers.

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt:**
torch
pandas
numpy
scikit-learn
matplotlib


---

## Reproducibility

All random seeds are fixed at the top of each script:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```



