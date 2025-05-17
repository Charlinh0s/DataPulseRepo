import pandas as pd
import pandas_ta as ta

# 1. LOAD & CLEAN RAW FILES

# --- 1. Cargar archivos -------------------------------------------------
df_D1  = pd.read_csv('forex_data/EURUSD_D1.csv',  delimiter='\t')
df_H4  = pd.read_csv('forex_data/EURUSD_H4.csv',     delimiter='\t')
df_H1  = pd.read_csv('forex_data/EURUSD_H1.csv',     delimiter='\t')
df_M15 = pd.read_csv('forex_data/EURUSD_M15.csv',    delimiter='\t')

# --- 2. Limpiar y estandarizar cada uno ---------------------------------

## a) Diario -------------------------------------------------------------
df_D1 = df_D1.rename(columns={
    '<DATE>'   : 'Date',
    '<OPEN>'   : 'Open',
    '<HIGH>'   : 'High',
    '<LOW>'    : 'Low',
    '<CLOSE>'  : 'Close',
    '<TICKVOL>': 'TickVolume'
})
df_D1['Datetime'] = pd.to_datetime(df_D1['Date'] + ' 00:00')
df_D1 = df_D1.drop(columns=['Date','<VOL>','<SPREAD>'])
df_D1 = df_D1.set_index('Datetime').sort_index()

## b) 4-Horas ------------------------------------------------------------
df_H4 = df_H4.rename(columns={
    '<DATE>'   : 'Date',
    '<TIME>'   : 'Time',
    '<OPEN>'   : 'Open',
    '<HIGH>'   : 'High',
    '<LOW>'    : 'Low',
    '<CLOSE>'  : 'Close',
    '<TICKVOL>': 'TickVolume'
})
df_H4['Datetime'] = pd.to_datetime(df_H4['Date'] + ' ' + df_H4['Time'])
df_H4 = df_H4.drop(columns=['Date', 'Time','<VOL>','<SPREAD>'])
df_H4 = df_H4.set_index('Datetime').sort_index()

## c) 1-Hora -------------------------------------------------------------
df_H1 = df_H1.rename(columns={
    '<DATE>'   : 'Date',
    '<TIME>'   : 'Time',
    '<OPEN>'   : 'Open',
    '<HIGH>'   : 'High',
    '<LOW>'    : 'Low',
    '<CLOSE>'  : 'Close',
    '<TICKVOL>': 'TickVolume'
})
df_H1['Datetime'] = pd.to_datetime(df_H1['Date'] + ' ' + df_H1['Time'])
df_H1 = df_H1.drop(columns=['Date', 'Time','<VOL>','<SPREAD>'])
df_H1 = df_H1.set_index('Datetime').sort_index()

## d) 15-Min -------------------------------------------------------------
df_M15 = df_M15.rename(columns={
    '<DATE>'   : 'Date',
    '<TIME>'   : 'Time',
    '<OPEN>'   : 'Open',
    '<HIGH>'   : 'High',
    '<LOW>'    : 'Low',
    '<CLOSE>'  : 'Close',
    '<TICKVOL>': 'TickVolume'
})
df_M15['Datetime'] = pd.to_datetime(df_M15['Date'] + ' ' + df_M15['Time'])
df_M15 = df_M15.drop(columns=['Date', 'Time','<VOL>','<SPREAD>'])
df_M15 = df_M15.set_index('Datetime').sort_index()

# --- 3. (Opcional) elimina índices duplicados ---------------------------
df_D1  = df_D1[~df_D1.index.duplicated()]
df_H4  = df_H4[~df_H4.index.duplicated()]
df_H1  = df_H1[~df_H1.index.duplicated()]
df_M15 = df_M15[~df_M15.index.duplicated()]

# Daily SMA‑200
df_D1['SMA200_D1'] = ta.sma(df_D1['Close'], length=200)
# 4‑Hour EMA‑50
df_H4['EMA50_H4'] = ta.ema(df_H4['Close'], length=50)
# 1‑Hour EMA‑50
df_H1['EMA50_H1'] = ta.ema(df_H1['Close'], length=50)
# 15‑Minute EMA‑20 & EMA‑50
df_M15['EMA20_M15'] = ta.ema(df_M15['Close'], length=20)
df_M15['EMA50_M15'] = ta.ema(df_M15['Close'], length=50)
# EMA20/EMA50 cross & crossunder on M15
df_M15['cross_up']   = ta.cross(df_M15['EMA20_M15'], df_M15['EMA50_M15'])
df_M15['cross_down'] = ta.cross(df_M15['EMA50_M15'], df_M15['EMA20_M15'])

df_D1 = df_D1[df_D1.index >= df_M15.index[0]]
df_H4 = df_H4[df_H4.index >= df_M15.index[0]]
df_H1 = df_H1[df_H1.index >= df_M15.index[0]]

df = df_M15.join(df_D1['SMA200_D1'], how='left')
df['SMA200_D1'] = df['SMA200_D1'].ffill()

df = df.join(df_H4['EMA50_H4'], how='left')
df['EMA50_H4'] = df['EMA50_H4'].ffill()

df = df.join(df_H1['EMA50_H1'], how='left')
df['EMA50_H1'] = df['EMA50_H1'].ffill()