# %% [markdown]
# This notebook is for providing the data and preprocess them for good use. 
# the data comes in a 10-minutes-mean with many differrent variables and goal is to have a one-hour mean and only the dtata for the temperature and the solar irradiance. 
# 
# 
# 

# %%
# import needed packages
import os 
import glob
import pandas as pd

# %%
#load data
files=sorted(glob.glob(os.path.join('./data/*/*/*msgg10m.dat')))
print(f'founded data: {len(files)} ')

Temp='AirTC_2_Avg'  
Rad ='SWUpper_Avg'
frames=[]


for filepath in files:
    df=pd.read_csv(filepath, skiprows=[0, 2, 3], header=0, quotechar='"', na_values=["NAN", "nan", ""] )

    df.columns = df.columns.str.strip().str.replace('\r', '')
    
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"].str.strip())
    df = df.set_index("TIMESTAMP")

    frames.append(df[[Temp, Rad]])
    
    #print(df)

    #print(frames)


data = pd.concat(frames).sort_index()
data = data[~data.index.duplicated(keep="first")]

#print(len(data))


# resample into hourly data
data_hourly = data.resample("1h").mean()
data_hourly[Rad] = data_hourly[Rad].clip(lower=0) #set negative values of radiation to 0

data_hourly = data_hourly.dropna()
#print(data_hourly)


print(f" {len(data_hourly)}")
print(f"time range: {data_hourly.index[0]} bis {data_hourly.index[-1]}")
print(f"temeprature range: {data_hourly[Temp].min():.1f} bis {data_hourly[Temp].max():.1f} °C")
print(f"Radiance range:  {data_hourly[Rad].min():.1f} bis {data_hourly[Rad].max():.1f} W/m²")



os.makedirs(os.path.dirname('./data_processed.csv'), exist_ok=True)
data_hourly.to_csv('data_processed.csv')
print(f"\n saved: {'data_processed.csv'}")




