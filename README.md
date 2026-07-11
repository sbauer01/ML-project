# Temperature Forecasting with a Feed-Forward Neural Network

This project was developed as **final project** of the course *Machine learning for the Earth System* at the University to Cologne in th summer term 2026. It implements a Feed-Forward Neural Network (FFN) in PyTorch to forecast air temperature one hour ahead using meteorological time series data.

---

## Project Overview

The model is trained on two input variables — air temperature and solar irradiation — measured at the weather station of the Institute of Metorology and Geohysics of the UNiversity to Cologne at the UniSport Cologne. The evaluation focuses exclusively on temperature prediction accuracy.

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

### Best and worst predicted days

To evaluate the range of accuracy of the model, the errors of the three best and the three worst predictedt days were analyzed. For the accuracy, a threhhold of 0.5°C were chosen. 


**Best predicted days:**

| Date       | MAE (°C) | RMSE (°C) | Max Error (°C) | Accuracy (< 0.5°C) |
|------------|----------|-----------|----------------|---------------------|
| 2026-05-06 | 0.27     | 0.30      | 0.52           | 95.83 %             |
| 2026-05-15 | 0.27     | 0.31      | 0.69           | 91.67 %             |
| 2026-04-13 | 0.26     | 0.34      | 0.88           | 87.50 %             |

**Worst predicted days:**

| Date       | MAE (°C) | RMSE (°C) | Max Error (°C) | Accuracy (< 0.5°C) |
|------------|----------|-----------|----------------|---------------------|
| 2026-03-27 | 0.65     | 0.73      | 1.26           | 33.33 %             |
| 2026-04-22 | 0.76     | 0.98      | 2.74           | 33.33 %             |
| 2026-05-12 | 0.70     | 0.86      | 2.39           | 33.33 %             |


The range of the accuracy goes from 33.33% to 95.83%. These results show, that this should be improved to have a higher reliabiliy. 

### Day vs. Night Accuracy

The model was evaluated separately for daytime and nighttime to assess performance under different radiation conditions. For that, the data was divided into daytime (7 am to 8pm) and nightime (8pm to 7am). The model showed these results: 

| Period | MAE (°C) | MSE (°C) | RMSE (°C) | Max Error (°C) | Accuracy (< 0.5°C) |
|--------|----------|----------|-----------|----------------|---------------------|
| Day    | 0.55     | 0.51     | 0.71      | 4.58           | 54.10 %             |
| Night  | 0.43     | 0.32     | 0.56      | 2.75           | 66.43 %             |

The performance of the model is better for at night. This could be, due to more stable temperatures and less influence of the solar irradiation variablity during nighttime.  



### Ablation Study

To assess the contribution of solar radiation as an input variable,  
an ablation test was conducted: the radiation values in the test set  
were replaced with random Gaussian noise (same mean and std as the  
original data). The resulting increase in MAE and RMSE quantifies  
the importance of radiation for temperature forecasting.

### Monthly Accuracy Distribution

For each month, the distribution of daily prediction accuracy (percentage of hours with error < 0.5°C) is anaylzed as a histogram with 10% bins and 25th, 50th, and 75th percentile markers.

| Month | 25th Percentile | Median | 75th Percentile |
|-------|----------------|--------|-----------------|
| March | 58.3 %         | 62.5 % | 66.7 %          |
| April | 54.2 %         | 62.5 % | 70.8 %          |
| May   | 41.7 %         | 58.3 % | 64.6 %          |

This shows, that April was the month with the best predicitons, followed by March and May, respectively. 

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



