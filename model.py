# %% [markdown]
# This notebook contains the main model for the project. 
# The idea is to create a machine learning model for a time series, taht predicts the temperature for a few days. 

# %% [markdown]
# The data is the measurements of the temperature and the incoming solar irradiance in an hourly-mean from the weather station at the UniSport in Cologne. The measurement periods were March, April, May of the years, 2024, 2025 and 2026. It was preprocessed from 10min mean to one hour mean and to have only the temperature and the radiation data. 
# 

# %%
#!pip3 install -r requirements.txt

# %%
# Import the needed modules

import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import random
from sklearn.preprocessing import StandardScaler

# %%
#define the randomness, so that we always have the same results

seed=23

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic=True


# %%
# Read in the preprocessed data

df=pd.read_csv('data_processed.csv', parse_dates=['TIMESTAMP'])

df

# %%
#normalize data

scale=MinMaxScaler()

df[['AirTC_2_Avg', 'SWUpper_Avg']]=scale.fit_transform(df[['AirTC_2_Avg', 'SWUpper_Avg']])

with open ('scale.pkl', 'wb') as f:
    pickle.dump(scale, f)
 

print(df[['AirTC_2_Avg', 'SWUpper_Avg']].describe())

# %%
#split the data in train und test data

train=df[df['TIMESTAMP'].dt.year.isin([2024, 2025])]
test= df[df['TIMESTAMP'].dt.year==2026]

print(len(train))
print(len(test))

# %%
#sliding window

window_size=24  #size of the window
 
def sequences(data):
    X, y=[],[]
    values=data[['AirTC_2_Avg','SWUpper_Avg']].values

    for i in range(len(values)-window_size):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size][0])

    return np.array(X), np.array(y)

X_train, y_train= sequences(train)
X_test, y_test=sequences(test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# %%
class WeatherData(Dataset):
    def __init__(self, X, y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return(len(self.X))
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

train_dataset=WeatherData(X_train, y_train)
test_dataset=WeatherData(X_test, y_test)

batch_size=32

train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(test_loader))
print(len(train_loader))


# %%
class FFN(nn.Module):
    def __init__(self):
        super (FFN, self).__init__()

        self.model=nn.Sequential(nn.Linear(24*2,64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    
    def forward(self,x):
        x=x.view(x.size(0),-1)
        return self.model(x).squeeze(1)
    

model=FFN()
print(model)


# %%
crit=nn.MSELoss() #mean squared-error--> how 'wrong' is model?

optimizer=optim.Adam(model.parameters(), lr=0.001)  #learning rate


run=50 #

for r in range(run):
    model.train()
    total_loss=0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions=model(X_batch)
        loss=crit(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    avg_loss=total_loss/len(train_loader)

    if (r+1)%10==0:
        print((r+1)/run, avg_loss)


# %% [markdown]
# ## Comparing the model results with the measurements

# %%
model.eval()
predictions=[]
actuals=[]

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        pred=model(X_batch)
        predictions.extend(pred.numpy())
        actuals.extend(y_batch.numpy())


predictions=np.array(predictions)
actuals=np.array(actuals)


#Mean Absolute error
mae=np.mean(np.abs(predictions-actuals))

#Mean squared error
mse=np.mean((predictions-actuals)**2)


print(f'MAE: {mae}')
print(f'MSE: {mse}')


# %%
# now convert back to °C for an easier interpretation

with open('scale.pkl', 'rb') as f:
    scale=pickle.load(f)

#transform predictions and measurements back
pred_rescaled=scale.inverse_transform(np.column_stack([predictions, np.zeros(len(predictions))]))[:,0]

actual_rescaled=scale.inverse_transform(np.column_stack([actuals, np.zeros(len(actuals))]))[:,0]



#Error in °C
mae_c=np.mean(np.abs(pred_rescaled-actual_rescaled))
mse_c=np.mean((pred_rescaled-actual_rescaled)**2)
rmse_c=np.sqrt(mse_c) #root mean squared error


print(f'MAE: {mae_c} °C')
print(f'MSE: {mse_c} °C')
print(f'RMSE: {rmse_c} °C')

# %%
#plot the model forecast vs the real measurements for specific days to see the differences

#for better visualisation: real dates instead of hours
test_timestamps = test['TIMESTAMP'].iloc[window_size:].values


# first:  5 days in a row = 120 hours, starting on first model data
n=120


plt.figure(figsize=(15, 6))
plt.plot(test_timestamps[:n], actual_rescaled[:n], label='measurements', color='blue') #unit: °C
plt.plot(test_timestamps[:n],pred_rescaled [:n], label='model prediction', color='red')
plt.xlabel('hours')
plt.ylabel('Temperature [°C]')
plt.legend()
plt.grid()
plt.show




# %%
#Lets go into detail, an look, which dates (per day) were quite good predicted and which ones not:

#build Dataframe
results=pd.DataFrame({'timestamp':test_timestamps, 'measurements':actual_rescaled, 'model prediction':pred_rescaled, 'error absolute':np.abs(pred_rescaled-actual_rescaled)})

#date as own coloumn

results['date']=pd.to_datetime(results['timestamp']).dt.date

#define threshold for good predicitons. above, the predicitons are too far away from measurements
threshold=0.5 #unit:°C

daily=results.groupby('date').agg(
    mae=('error absolute', 'mean'),  #mean error
    mse= ("error absolute", lambda x: np.mean(x**2)),  #mean squared error
    rmse= ("error absolute", lambda x: np.sqrt(np.mean(x**2))),
    max_error=('error absolute', 'max'), #biggest error
    good_pred=('error absolute', lambda x: (x<threshold).sum()), #amount of 'good' predictions
    number_pred=('error absolute', 'count')) #number of predicitions, should be 24 -> 24 hours a day
    

daily['accuracy']=(daily['good_pred']/daily['number_pred']*100).round(2)

print(daily)


# %%
#Maximum and Minimum of the accuracy -> which 3 days were predicted best and worst, respectively

max_days=daily.nlargest(3, 'accuracy')
min_days=daily.nsmallest(3, 'accuracy')



print('Best days:')
print(max_days)
print('Worst days:')
print(min_days)




# %%
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 30), constrained_layout=True)

# Beste 3 Tage
for ax, (date, row), title in zip(
    [ax1, ax2, ax3],
    max_days.iterrows(),
    ["Day with best prediction", "Day with 2nd best prediction", "Day with 3rd best prediction"]
):
    mask     = results["date"] == date
    day_data = results[mask]

    ax.plot(day_data["timestamp"], day_data["measurements"],     label="measurements",    color="blue")
    ax.plot(day_data["timestamp"], day_data["model prediction"], label="model prediction", color="red")
    ax.set_xlabel("Time (date (mm-dd) and hour)")
    ax.set_ylabel("Temperature [°C]")
    ax.set_ylim(0, 22 )
    ax.set_title(f"{title}: {date}  |  MAE: {row['mae']:.2f}°C  |  Accuracy: {row['accuracy']}%")
    ax.legend()
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.grid(True, which="major", linestyle="-",  linewidth=0.8)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.4)

# Schlechteste 3 Tage
for ax, (date, row), title in zip(
    [ax4, ax5, ax6],
    min_days.iterrows(),
    ["Day with worst prediction", "Day with 2nd worst prediction", "Day with 3rd worst prediction"]
):
    mask     = results["date"] == date
    day_data = results[mask]

    ax.plot(day_data["timestamp"], day_data["measurements"],     label="measurements",    color="blue")
    ax.plot(day_data["timestamp"], day_data["model prediction"], label="model prediction", color="red")
    ax.set_xlabel("Time (date (mm-dd) and hour)")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{title}: {date}  |  MAE: {row['mae']:.2f}°C  |  Accuracy: {row['accuracy']}%")
    ax.set_ylim(0, 22)
    ax.legend()
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.grid(True, which="major", linestyle="-",  linewidth=0.8) 
    ax.grid(True, which="minor", linestyle="--", linewidth=0.4)

plt.show()

# %%
#now lets look, if the predicition if the night or of the day is better
#Divide the day: daytime: 7am to 8 pm, nighttime: 8pm to 7am

#extract gour from timestamp
results['hour']=pd.to_datetime(results['timestamp']).dt.hour


results['period']=results['hour'].apply(lambda h:'day' if 7<=h<20 else 'night')

period_analysis = results.groupby('period').agg(
    mae=('error absolute', 'mean'),
    mse= ("error absolute", lambda x: np.mean(x**2)), 
    rmse= ("error absolute", lambda x: np.sqrt(np.mean(x**2))),
    max_error=('error absolute', 'max'),
    good_pred=('error absolute', lambda x: (x<threshold).sum()), 
    number_pred=('error absolute', 'count')
).reset_index()


period_analysis['accuracy']=(period_analysis['good_pred']/period_analysis['number_pred']*100).round(2)


print(period_analysis)

# %%
#monthly frequency distribution

daily["month"] = pd.to_datetime(daily.index).month
daily['month_name']=daily['month'].map({3:'March', 4:'April', 5:'May'})

#define accuracy bins
bins   = list(range(0, 110, 10))
labels = [f"{i}-{i+10}%" for i in range(0, 100, 10)]

daily["accuracy_bin"] = pd.cut(daily["accuracy"], bins=bins, labels=labels)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for ax, (month_name, group) in zip(axes, daily.groupby("month_name")):
    counts = group["accuracy_bin"].value_counts().reindex(labels, fill_value=0)
    ax.bar(labels, counts, color="steelblue", edgecolor="white")

    # Perzentile berechnen
    p25 = group["accuracy"].quantile(0.25)
    p50 = group["accuracy"].quantile(0.50)
    p75 = group["accuracy"].quantile(0.75)

    # Vertikale Linien – müssen auf Bin-Position gemappt werden
    for p, color, label in zip(
        [p25, p50, p75],
        ["green", "red", "orange"],
        [f"25th percentile ({p25:.1f}%)", f"Median ({p50:.1f}%)", f"75th percentile ({p75:.1f}%)"]
    ):
        # Bin-Position berechnen (jeder Bin ist 10% breit)
        bin_pos = p / 10 - 0.5  # Mitte des jeweiligen Bins
        ax.axvline(x=bin_pos, color=color, linestyle="--", linewidth=1.5, label=label)

    ax.set_title(month_name)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of days")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linewidth=0.5)
    ax.legend(fontsize=8)

plt.suptitle("Accuracy Distribution per Month", fontsize=14)
plt.show()


# %% [markdown]
# ## Ablation study
# 
# In this ablation study, the radiation input gets randomised to see, what impact this inputs has to our model prediction for the temperature. 
# 
# **Hypothesis:** If the input of the model is based on real temperature measuremnets and randomized irradiation values, the model will performe worse, since the range of real data is less and the irradiation values are more worthless. 

# %%
#Copy the test data
X_test_ablation=X_test.copy()

#replace radiation values with random values 
#Gaußiand distribution as the radiation values
rad_mean=X_test[:,:,1].mean()
rad_std=X_test[:,:,1].std()

np.random.seed(42)
X_test_ablation[:,:,1] = np.random.normal(loc=rad_mean, scale=rad_std, size=X_test[:, :, 1].shape).clip(0, 1)  #[0,1], because normalized

# Inference with random data
model.eval()
predictions_ablation = []

with torch.no_grad():
    for i in range(0, len(X_test_ablation), 32):
        batch=torch.tensor(X_test_ablation[i:i+32], dtype=torch.float32)
        pred=model(batch)
        predictions_ablation.extend(pred.numpy())

predictions_ablation=np.array(predictions_ablation)

#go back to °C
pred_ablation_rescaled = scale.inverse_transform(
    np.column_stack([predictions_ablation, np.zeros(len(predictions_ablation))]))[:, 0]

#Compare the results with the real measurements
mae_normal=np.mean(np.abs(pred_rescaled-actual_rescaled))
mae_ablation=np.mean(np.abs(pred_ablation_rescaled-actual_rescaled))

mse_normal=np.mean((pred_rescaled-actual_rescaled)**2)
mse_ablation=np.mean((pred_ablation_rescaled-actual_rescaled)**2)

rmse_normal=np.sqrt(mse_normal)
rmse_ablation=np.sqrt(mse_ablation)




print(f"MAE with measurement data: {mae_normal:.2f} °C")
print(f"MAE with random radiation: {mae_ablation:.2f} °C")


print(f"Difference: {mae_ablation-mae_normal:.2f} °C")

print(f"{'':20} {'Measurement':>10} {'Random':>10} {'Difference':>10}")
print(f"{'MAE (°C)':20} {mae_normal:>10.2f} {mae_ablation:>10.2f} {mae_ablation - mae_normal:>+10.2f}")
print(f"{'MSE (°C)':20} {mse_normal:>10.2f} {mse_ablation:>10.2f} {mse_ablation - mse_normal:>+10.2f}")
print(f"{'RMSE (°C)':20} {rmse_normal:>10.2f} {rmse_ablation:>10.2f} {rmse_ablation - rmse_normal:>+10.2f}")

# %% [markdown]
# **Result:** The results verify the hypothesis. The errors for the model with randomized irradiaiton values are much higher (up to five times) than the errors for the real measurements. 
# This means, that a bigger input (more different variables) leads to a higher performance of the model. 

# %% [markdown]
# ## Evaluation - Persistence Baseline
# 
# This part is a simple baseline experiment, where we say, that the temperature in the next hour is the same as now. So the question is, if our model is better than no model. 
# 
# **Hypothesis:** The trained model should show smaller error values, since it can perform better for periods of rapid remperature changes, where simply repeating the current values leads to larger errors.

# %%


# %%
#persistence value
persistence_pred = X_test[:, -1, 0]  #last time step, temperature

# Convert back to °C
persistence_rescaled = scale.inverse_transform(
    np.column_stack([persistence_pred, np.zeros(len(persistence_pred))])
)[:, 0]

# Error calculation
mae_persistence  = np.mean(np.abs(persistence_rescaled - actual_rescaled))
mse_persistence  = np.mean((persistence_rescaled - actual_rescaled)**2)
rmse_persistence = np.sqrt(mse_persistence)


print(f"{'':20} {'FFN':>10} {'Persistence':>12} {'Difference':>10}")
print(f"{'MAE (°C)':20} {mae_c:>10.2f} {mae_persistence:>12.2f} {mae_c - mae_persistence:>+10.2f}")
print(f"{'MSE (°C²)':20} {mse_c:>10.2f} {mse_persistence:>12.2f} {mse_c - mse_persistence:>+10.2f}")
print(f"{'RMSE (°C)':20} {rmse_c:>10.2f} {rmse_persistence:>12.2f} {rmse_c - rmse_persistence:>+10.2f}")

# %% [markdown]
# **Result:** The hypothesis was verified by the results. The performance of the model is worse for the perisstence baseline. This confirmed that the model  has learned meaningful patterns beyond simply repeating the current temperature. 
# 


