import pandas as pd
import pandas_ta as ta

# --- 1. Función para verificar cercanía al SMA200 ------------------------
def is_near(price, ema, margin=0.1):
    return abs(price - ema) / ema <= margin

# --- 2. Cargar archivos -------------------------------------------------
df_D1  = pd.read_csv('forex_data/EURUSD_D1.csv',  delimiter='\t')
df_H4  = pd.read_csv('forex_data/EURUSD_H4.csv', delimiter='\t')
df_H1  = pd.read_csv('forex_data/EURUSD_H1.csv', delimiter='\t')
df_M15 = pd.read_csv('forex_data/EURUSD_M15.csv', delimiter='\t')

# --- 3. Limpiar y estandarizar cada uno ---------------------------------

def preprocess_df(df, timeframe):
    df = df.rename(columns={
        '<DATE>'   : 'Date',
        '<TIME>'   : 'Time',
        '<OPEN>'   : 'Open',
        '<HIGH>'   : 'High',
        '<LOW>'    : 'Low',
        '<CLOSE>'  : 'Close',
        '<TICKVOL>': 'TickVolume'
    })
    if timeframe == 'D1':
        df['Datetime'] = pd.to_datetime(df['Date'] + ' 00:00')
        df = df.drop(columns=['Date','<VOL>','<SPREAD>'])
    else:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.drop(columns=['Date','Time','<VOL>','<SPREAD>'])
    df = df.set_index('Datetime').sort_index()
    return df[~df.index.duplicated()]

df_D1  = preprocess_df(df_D1, 'D1')
df_H4  = preprocess_df(df_H4, 'H4')
df_H1  = preprocess_df(df_H1, 'H1')
df_M15 = preprocess_df(df_M15, 'M15')

# --- 4. Indicadores técnicos --------------------------------------------

for df_tf, name in [(df_D1, 'D1'), (df_H4, 'H4'), (df_H1, 'H1'), (df_M15, 'M15')]:
    # RSI
    df_tf[f'RSI_{name}'] = ta.rsi(df_tf['Close'], length=14)
    # MACD
    macd = ta.macd(df_tf['Close'], fast=12, slow=26, signal=9)
    df_tf[f'MACD_{name}'] = macd['MACD_12_26_9']
    df_tf[f'MACD_signal_{name}'] = macd['MACDs_12_26_9']
    # EMAs
    df_tf[f'EMA20_{name}'] = ta.ema(df_tf['Close'], length=20)
    df_tf[f'EMA50_{name}'] = ta.ema(df_tf['Close'], length=50)
    # Above EMA50
    df_tf[f'above_EMA50_{name}'] = (df_tf['Close'] > df_tf[f'EMA50_{name}']).astype(int)
    # Cruce de EMAs
    df_tf[f'cross_up_{name}'] = ta.cross(df_tf[f'EMA20_{name}'], df_tf[f'EMA50_{name}'])
    df_tf[f'cross_down_{name}'] = ta.cross(df_tf[f'EMA50_{name}'], df_tf[f'EMA20_{name}'])

# SMA200 solo en diario
df_D1['SMA200_D1'] = ta.sma(df_D1['Close'], length=200)
df_D1['near_SMA200_D1'] = is_near(df_D1['Close'], df_D1['SMA200_D1'])

# --- 5. Recortar los dataframes a partir de df_M15 ----------------------
df_D1 = df_D1[df_D1.index >= df_M15.index[0]]
df_H4 = df_H4[df_H4.index >= df_M15.index[0]]
df_H1 = df_H1[df_H1.index >= df_M15.index[0]]

# --- 6. Combinar en el dataframe principal (df_M15) ---------------------
df = df_M15.copy()

df = df.join(df_H1[[f for f in df_H1.columns if '_H1' in f]], how='left')
df = df.join(df_H4[[f for f in df_H4.columns if '_H4' in f]], how='left')
df = df.join(df_D1[[f for f in df_D1.columns if '_D1' in f]], how='left')

# --- 7. Rellenar valores nulos hacia adelante ---------------------------
df = df.ffill()

# --- 8. Crear columna Target: sube (1) o baja (0) después de un día -----
future_close = df['Close'].shift(-96)  # 1 día = 96 velas de 15 minutos
df['Target'] = (future_close > df['Close']).astype(int)

df = df.dropna()
df.info()
df['cross_up_M15'].value_counts()