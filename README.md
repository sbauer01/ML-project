# Temperature Forecasting with a Feed-Forward Neural Network

This project was developed as **final project** of the course *Machine learning for the Earth System* at the University of Cologne in the summer term 2026. It implements a Feed-Forward Neural Network (FFN) in PyTorch to forecast air temperature one hour ahead using meteorological time series data.

---

## Project Overview

The model is trained on two input variables — air temperature and solar irradiation — measured at the weather station of the Institute of Geophysics and Meteorology at the University of Cologne at the UniSport Cologne. The evaluation focuses exclusively on temperature prediction accuracy.

**Training data:** March, April, May 2024 and March, April, May 2025  
**Test data:** March, April, May 2026  
**Forecast horizon:** One step ahead (1 hour)  
**Input window:** 24 hours  

---

## Repository Structure
ML-project/

├── data_preprocessing.ipynb   # Jupyter notebook: data loading, resampling, normalization
├── model.ipynb                # Jupyter notebook: model, training, evaluation and results
├── data_processed.csv         # Preprocessed hourly data (temperature + radiation)
├── scale.pkl                  # Fitted MinMaxScaler for inverse transformation
├── requirements.txt
├── .gitignore
└── README.md

> **Note:** Raw data files (`data/`) are not included in this repository due to file size. See **Data** section below.

---

## Data

The data was measured at the weather station at the UniSport Cologne, in 1-minute intervals and downloaded in a 10-minute mean values.

The preprocessing extracts the temperature data and the incoming solar irradiation measurements and calculates the hourly mean values of it. 

| Variable | Column | Unit |
|---|---|---|
| Air temperature (2m) | `AirTC_2_Avg` | °C |
| Shortwave radiation (upward) | `SWUpper_Avg` | W/m² |

Raw files follow the naming convention `YYYYMMDD_msgg10m.dat` and are  organized in the folder structure `data/MM/DD/`.

The data is visible on the DataBrowser of the weather station: https://atmos.meteo.uni-koeln.de/~citystation/dataBrowser/dataBrowser1.html?site=CITYSTATION&date=0&UpperLeft=Temperatur_K%C3%B6ln and can be requested under wetter-station@uni-koeln.de.



## Model

```
Input:  24 timesteps × 2 variables = 48 values
→ Linear(48 → 64) + ReLU
→ Linear(64 → 32) + ReLU
→ Linear(32 → 1)
Output: predicted temperature at t+1
```
---

## Training

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr = 0.001)
- **Batch size:** 32
- **Epochs:** 50
- **Seed:** 23 (for reproducibility)

---

## Results

### Overall model performance

The whole accuracy of the model was analyzed. Therefore, the Mean absolute error (MAE), the mean squared error (MSE), and the root mean square error (RMSE) were calculated.

| Metric | Value |
|---|---|
| MAE | 0.49 °C |
| MSE | 0.42 °C² |
| RMSE | 0.65 °C |



The MAE of 0.49°C indicates that the model deviates from the measured temperature by less than 0.5°C on average. The RMSE of 0.65°C is slightly higher, which means there are occasional larger errors that pull the value up — but overall the model performs consistently well.


### Best and worst predicted days

To evaluate the range of accuracy of the model, the errors of the three best and the three worst predicted days were analyzed. For the accuracy, a threshold of 0.5°C was chosen, as this value represents a meteorologically meaningful 
level of precision — deviations below 0.5°C are generally considered negligible for practical temperature forecasting applications and are within the typical measurement uncertainty of temperature sensors.

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


The range of the accuracy goes from 33.33% to 95.83%. These results show that this should be improved to have a higher reliability. 

### Day vs. Night Accuracy

The model was evaluated separately for daytime and nighttime to assess performance under different radiation conditions. For that, the data was divided into daytime (7 am to 8 pm) and nighttime (8 pm to 7 am). The model showed these results: 

| Period | MAE (°C) | MSE (°C²) | RMSE (°C) | Max Error (°C) | Accuracy (< 0.5°C) |
|--------|----------|----------|-----------|----------------|---------------------|
| Day    | 0.55     | 0.51     | 0.71      | 4.58           | 54.10 %             |
| Night  | 0.43     | 0.32     | 0.56      | 2.75           | 66.43 %             |

The performance of the model is better at night. This could be due to more stable temperatures and less influence of the solar irradiation variability during nighttime.  


### Monthly Accuracy Distribution

For each month, the distribution of daily prediction accuracy (percentage of hours with error < 0.5°C) is analyzed as a histogram with 10% bins and 25th, 50th, and 75th percentile markers.

| Month | 25th Percentile | Median | 75th Percentile |
|-------|----------------|--------|-----------------|
| March | 58.3 %         | 62.5 % | 66.7 %          |
| April | 54.2 %         | 62.5 % | 70.8 %          |
| May   | 41.7 %         | 58.3 % | 64.6 %          |


March shows the most consistent performance with the narrowest spread between the 25th and 75th percentile (8.4%), indicating stable and reliable predictions throughout the month.
April achieves the same median as March but with a wider spread (16.6%), suggesting higher day-to-day variability — likely due to more dynamic spring weather patterns.
May has both the lowest median (58.3%) and the widest overall distribution, with some days falling below 20% accuracy. This points to more complex meteorological conditions in late spring, such as convective events and rapid temperature changes, which are  harder for the model to capture.



### Ablation Study

**Hypothesis**: If the model just uses measured temperature values and gets randomised irradiation values as input, the performance will be worse, since the model's knowledge of reality is less than if there are also the measured irradiation values as input. 

To assess the contribution of solar radiation as an input variable, an ablation test was conducted: the radiation values in the test set were replaced with random Gaussian noise (same mean and std as the original data). The model was then evaluated on these corrupted inputs without retraining.

| Metric   | Original | Random Radiation | Difference |
|----------|----------|------------------|------------|
| MAE (°C) | 0.49     | 1.20             | +0.71      |
| MSE (°C²)| 0.42     | 2.23             | +1.81      |
| RMSE (°C)| 0.65     | 1.49             | +0.84      |


Replacing the solar radiation with random noise roughly doubles the MAE and increases the MSE by a factor of five. This confirms that the model has learned to meaningfully use radiation as an input feature — without it, prediction accuracy degrades significantly. So, the hypothesis was verified. 


### Persistence Baseline

To evaluate the model, a comparison to a persistence baseline was made. As a baseline, a persistence model was used: the predicted temperature for the next hour is simply the current temperature. With that, the question if the trained model performs better than no model can be answered.

**Hypothesis:** The FFN model will outperform the persistence baseline, since during periods of rapid temperature change, simply repeating the current value will lead to larger errors.


| Metric    | FFN  | Persistence | Difference |
|-----------|------|-------------|------------|
| MAE (°C)  | 0.49 | 0.87        | -0.38      |
| MSE (°C²) | 0.42 | 1.29        | -0.87      |
| RMSE (°C) | 0.65 | 1.14        | -0.49      |

The hypothesis was verified. The FFN outperforms the persistence baseline across all metrics, reducing the MAE by 0.38°C and the RMSE by 0.49°C. This confirms that the model has learned meaningful temporal patterns from the 24-hour input window rather than simply memorizing the current state.


---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- torch
- pandas
- numpy
- scikit-learn
- matplotlib


---

## Reproducibility

All random seeds are fixed at the top of each script:

```python
random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
```

---



# Academic context
This project was developed as the final project for the course *Machine Learning for the Earth System (MLESS)* at the University of Cologne.

Instructor: Prof. Dr. Martin Schultz



## Author
Simone Bauer
Master's program: Physics of the Earth and Atmosphere
University of Cologne



