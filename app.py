
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly as plt
import plotly.graph_objects as go
import exchange_calendars as xcals
from datetime import date
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from numpy.linalg import multi_dot
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.optimize as sco
from numpy import *

############################## FUNCIONES #######################################
def download_data(tickers, start_date='2001-01-01', end_date=date.today().strftime('%Y-%m-%d')):
    data = yf.download(tickers, start=start_date, end=end_date)

    return data['Close']

def calcular_fechas(hoy: pd.Timestamp):
    # Obtén el calendario de la bolsa de México
    xmex = xcals.get_calendar("XMEX")

    # Si el día de la semana es lunes (0 en el sistema Python weekday()), retrocede 3 días
    if hoy.weekday() == 0:
        prev_business_day = hoy - pd.Timedelta(days=3)
    # De lo contrario, solo retrocede un día
    else:
        prev_business_day = hoy - pd.Timedelta(days=1)

    # Si el día calculado no es un día hábil, busca el día hábil más reciente
    if not xmex.is_session(prev_business_day):
        prev_business_day = xmex.previous_close(prev_business_day).to_pydatetime()

    ayer = prev_business_day

    # Crear un diccionario para almacenar los resultados
    resultado = {}

    # Mes hasta la fecha
    primer_dia_mes = xmex.date_to_session(hoy.replace(day=1), direction="next")
    if hoy == primer_dia_mes:
        # Si hoy es el primer día hábil del mes, toma el primer día hábil del mes anterior
        mes_anterior = hoy - pd.DateOffset(months=1)
        primer_dia_mes = xmex.date_to_session(mes_anterior.replace(day=1), direction="next")

    # Calcula los días hábiles entre el primer día del mes y hoy
    dias_habiles = len(xmex.sessions_in_range(primer_dia_mes, hoy))+1

    # Usa estos días hábiles para obtener la ventana de sesiones
    mes_hasta_fecha = xmex.sessions_window(hoy, -dias_habiles)

    # Año hasta la fecha
    primer_dia_año = xmex.date_to_session(hoy.replace(month=1, day=1), direction="next")
    if hoy == primer_dia_año:
        # Si hoy es el primer día hábil del año, toma el primer día hábil del año anterior
        año_anterior = hoy - pd.DateOffset(years=1)
        primer_dia_año = xmex.date_to_session(año_anterior.replace(month=1, day=1), direction="next")

    # Calcula los días hábiles entre el primer día del año y hoy
    dias_habiles = len(xmex.sessions_in_range(primer_dia_año, hoy))+1

    # Usa estos días hábiles para obtener la ventana de sesiones
    año_hasta_fecha = xmex.sessions_window(hoy, -dias_habiles)

    # Fecha de hace un mes
    hace_un_mes = hoy - pd.DateOffset(months=1)

    # Encuentra el día hábil más cercano en el pasado a hace_un_mes
    dia_habil_hace_un_mes = xmex.date_to_session(hace_un_mes, direction="previous")

    # Obtén todas las sesiones desde hace_un_mes hasta hoy
    ultimos_30_dias = xmex.sessions_in_range(dia_habil_hace_un_mes, hoy)

    # Fecha de hace tres meses
    hace_tres_meses = hoy - pd.DateOffset(months=3)

    # Encuentra el día hábil más cercano en el pasado a hace_tres_meses
    dia_habil_hace_tres_meses = xmex.date_to_session(hace_tres_meses, direction="previous")

    # Obtén todas las sesiones desde hace_tres_meses hasta hoy
    ultimos_90_dias = xmex.sessions_in_range(dia_habil_hace_tres_meses, hoy)

    # Fecha de hace seis meses
    hace_seis_meses = hoy - pd.DateOffset(months=6)

    # Encuentra el día hábil más cercano en el pasado a hace_seis_meses
    dia_habil_hace_seis_meses = xmex.date_to_session(hace_seis_meses, direction="previous")

    # Obtén todas las sesiones desde hace_seis_meses hasta hoy
    ultimos_180_dias = xmex.sessions_in_range(dia_habil_hace_seis_meses, hoy)

    # Fecha de hace un año
    hace_un_año = hoy - pd.DateOffset(years=1)

    # Encuentra el día hábil más cercano en el pasado a hace_un_año
    dia_habil_hace_un_año = xmex.date_to_session(hace_un_año, direction="previous")

    # Obtén todas las sesiones desde hace_un_año hasta hoy
    ultimos_365_dias = xmex.sessions_in_range(dia_habil_hace_un_año, hoy)

    resultado['mes_hasta_fecha'] = mes_hasta_fecha
    resultado['año_hasta_fecha'] = año_hasta_fecha
    resultado['ultimos_30_dias'] = ultimos_30_dias
    resultado['ultimos_90_dias'] = ultimos_90_dias
    resultado['ultimos_180_dias'] = ultimos_180_dias
    resultado['ultimos_365_dias'] = ultimos_365_dias

    return resultado

def anualizar_rendimiento(rendimiento_bruto, dias):
    rendimiento_anualizado = rendimiento_bruto / dias * 360
    return rendimiento_anualizado

def calcular_rendimiento_bruto(precio_inicio, precio_fin, dias):
    # Calcular el cambio porcentual en el precio
    cambio_pct = (precio_fin / precio_inicio) - 1

    # Calcular el rendimiento bruto
    rendimiento_bruto = cambio_pct
    return rendimiento_bruto

def calcular_rendimiento(precios, ventanas_de_tiempo, nombre_benchmark):
    rendimientos = []

    for periodo, ventana in ventanas_de_tiempo.items():
        # Obtén los precios de inicio y fin para la ventana de tiempo actual
        precio_inicio = precios.loc[ventana[0], nombre_benchmark]
        precio_fin = precios.loc[ventana[-1], nombre_benchmark]

        # Calcula el rendimiento bruto y anualizado
        rendimiento_bruto = calcular_rendimiento_bruto(precio_inicio, precio_fin, (ventana[-1] - ventana[0]).days)
        rendimiento_anualizado = anualizar_rendimiento(rendimiento_bruto, (ventana[-1] - ventana[0]).days)

        # Agrega el rendimiento a la lista de rendimientos
        rendimientos.append({
            'Periodo': periodo,
            'Rendimiento_bruto': rendimiento_bruto*100,
            'Rendimiento_anualizado': rendimiento_anualizado*100
        })

    # Convierte la lista de rendimientos en un dataframe
    df_rendimientos = pd.DataFrame(rendimientos)

    return df_rendimientos

#Descargamos datos
def download_datas(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

#Calculo de las estadisticas
def calculos_est(SR_weight,df_MXN,entero):
    returns = df_MXN.pct_change().fillna(0)#cambio porcentual
    n_sim=100000
    Rf = 0.05058 / 252
    lista_means,lista_std,lista_skw,lista_VaR_Param,lista_VaR_Hist=[],[],[],[],[]
    lista_VaR_MC,lista_CVaR,lista_SR,lista_Sortino,lista_Acum=[],[],[],[],[]

    if entero==1:
        final_returns=returns
    elif entero==0:
        returns=returns*SR_weight
        final_returns=pd.DataFrame(returns.sum(axis=1))

    for i in final_returns.columns:
        lista_means.append(np.mean(final_returns[i]))
        lista_std.append(np.std(final_returns[i]))
        lista_skw.append(final_returns[i].skew())
        lista_VaR_Param.append(norm.ppf(1-0.95,np.mean(final_returns[i]),np.std(final_returns[i])))
        lista_VaR_Hist.append(final_returns[i].quantile(0.05))
        lista_VaR_MC.append(np.percentile(np.random.normal(np.mean(final_returns[i]),np.std(final_returns[i]),n_sim),5))
        lista_CVaR.append(final_returns[final_returns[i]<=final_returns[i].quantile(0.05)][i].mean())
        if np.std(final_returns[i])!=0:
            lista_SR.append((np.mean(final_returns[i])-Rf)/np.std(final_returns[i]))
            lista_Sortino.append((np.mean(final_returns[i])-Rf)/final_returns[final_returns[i]<Rf][i].std())
        else:
            lista_SR.append(0)
            lista_Sortino.append(0)


    nombres_col=['Ticker','Media','Sesgo','Curtosis','VaR Parametrico','VaR Historico','VaR MonteCarlo','CVaR',
                'Sharpe Ratio','Sortino']
    lista_tickers=pd.DataFrame(final_returns.columns)
    lista_means=pd.DataFrame(lista_means)
    lista_std=pd.DataFrame(lista_std)
    lista_skw=pd.DataFrame(lista_skw)
    lista_VaR_Param=pd.DataFrame(lista_VaR_Param)
    lista_VaR_Hist=pd.DataFrame(lista_VaR_Hist)
    lista_VaR_MC=pd.DataFrame(lista_VaR_MC)
    lista_CVaR=pd.DataFrame(lista_CVaR)
    lista_SR=pd.DataFrame(lista_SR)
    lista_Sortino=pd.DataFrame(lista_Sortino)
    lista_Acum=(final_returns+1).cumprod()
    lista_Acum=pd.DataFrame(lista_Acum)


    junto=pd.concat([lista_tickers,lista_means,lista_std,lista_skw,lista_VaR_Param,lista_VaR_Hist,lista_VaR_MC,lista_CVaR,
                    lista_SR,lista_Sortino],axis=1)
    junto=junto.fillna(0)
    junto.columns=nombres_col
    junto.set_index('Ticker',drop=True,inplace=True)
    return junto,lista_Acum

# definimos una funcion de como comienza el portafolio
def portfolio_stats(weights):
  weights = array(weights)[:,newaxis]
  port_rets = weights.T @ array(returns.mean() * 252)[:,newaxis]
  port_vols = sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))

  return array([port_rets, port_vols, port_rets/port_vols]).flatten()

# definificion de funcion para maximizar sharpe ratio
def min_sharpe_ratio(weights):
  return -portfolio_stats(weights)[2]

# Funcion de minima varianza
def min_variance(weights):
    return portfolio_stats(weights)[1]**2

def min_volatility(weights):
    return portfolio_stats(weights)[1]

# Definimos funcion que de los retornos esperados dado los pesos
def returns_means(weights):
  port_rets2 = weights.T @ array(returns.mean() * 252)[:,newaxis]
  return port_rets2

################################################################################
# Descarga de la información de los activo
tickers = ['IEMB.MI', 'EWZ', 'STIP', "IVV", "IAU"]

activos=download_data(tickers)
activos = activos.dropna()

df_activos = activos.copy()

######################### Botones principales ##################################

# Opciones de navegación
st.sidebar.title("Navegación")
option = st.sidebar.radio("Seleccione una página", ["Activos", "Portafolios"])

################################################################################

if option == "Portafolios":
    portafolio = st.sidebar.selectbox(
        "Elige dependiendo la información que desees visualizar",
        ('Información','A', 'B', 'C') )


if option == "Portafolios":

    if portafolio == "Información":
        st.title("Optimización de Portafolios")
        st.markdown("A continuación podrá leer una descripción de lo que se desarrolla en cada inciso a elegir.")
        st.markdown("**Opción A:** Estadísticas de los Activos. Calculo usando los años 2010 a 2023 de los rendimientos diarios, media, sesgo, exceso de curtosis, VaR, CVaR, Sharpe Ratio, Sortino y Drawdownde de los 5 ETFs cistos en el área de activos")
        st.markdown("**Opción B:** Calculo de portafolios ́optimos usando los ETFs vistos en la parte de activos y dadas las restricciones: mínima volatilidad, máximo sharpe ratio y mínima volatilidad con objetivo de rendimiento de 10 % anual, usando rendimientos del 2010 a 2020.")
        st.markdown("**Opción C:** Tomándose en cuenta los portafolios óptimos que se calcularon en la opción B, el índice S&P500, el portafolio de pesos iguales y con datos del 2021 al 2023, se calcularan los rendimientos anuales, rendimientos acumulados, sesgo, exceso de curtosis, VaR, CVaR, Sharpe Ratio, Sortino, y Drawdown. Concluyendo donde hubiera sido mejor invertir tus recurso.")
        st.markdown("Obs. Todos los incisos devuelven valores en pesos mexicanos $")

    if portafolio == "A":
        st.title("Estadísticas de los activos")
        st.header("Precios por activo")
        st.markdown("Obs. Como se comento en el apartado de información todos los precios dados ya se encuentran en pesos mexicanos.")
        symbols = ['IVV', 'IEMB.MI', 'STIP', 'IAU', 'EWZ', 'MXN=X', 'EURMXN=X']

        df=download_datas(symbols,'2010-01-01',None)

        #Pasamos todos a MXN
        df_MXN = pd.DataFrame(columns= ['IVV', 'IEMB.MI', 'STIP', 'IAU', 'EWZ'])
        df_MXN['IVV'] = df['IVV']*df['MXN=X']
        df_MXN['IEMB.MI'] = df['IEMB.MI']*df['EURMXN=X']
        df_MXN['STIP'] = df['STIP']*df['MXN=X']
        df_MXN['IAU'] = df['IAU']*df['MXN=X']
        df_MXN['EWZ'] = df['EWZ']*df['MXN=X']
        df_MXN = df_MXN.dropna()
        st.write(df_MXN)

        st.header("Rendimientos")
        df_MXNN = df_MXN.pct_change().dropna()
        st.write(df_MXNN)

        st.header("Estadisticas")
        weight_b=[1,1,1,1,1]
        inciso_b,jeje=calculos_est(weight_b,df_MXN,1)
        st.write(inciso_b)

        st.header("Drawdown")

        #Se refiere a cuánto ha caído una cuenta desde su punto máximo hasta su punto mínimo.
        # Calcular la serie acumulativa de rendimientos
        df_acumulativo = (1+df_MXNN).cumprod()

        # Calcular el pico acumulativo previo en cada punto del tiempo
        df_pico = df_acumulativo.cummax()

        # Calcular la serie de drawdowns
        df_drawdown = (df_acumulativo-df_pico)/df_pico

        # Encontrar el drawdown máximo
        df_max_drawdown = df_drawdown.min()
        df_max_drawdown.columns = ['%']

        st.markdown(f"En su peor momento, los activos experimentaron las siguientes caídas en porcentaje, respecto su pico anterior")
        st.write(df_max_drawdown)

        st.markdown(f"Los últimos valores de la serie de drawdowns por activos, son:")
        st.write(df_drawdown.tail())

    elif portafolio == "C":
        st.title("Back Testing")

        symbols = ['IVV', 'IEMB.MI', 'STIP', 'IAU', 'EWZ', 'MXN=X', 'EURMXN=X']

        df=download_datas(symbols,'2021-01-01',None)

        #Pasamos todos a MXN
        df_MXN = pd.DataFrame(columns= ['IVV', 'IEMB.MI', 'STIP', 'IAU', 'EWZ'])
        df_MXN['IVV'] = df['IVV']*df['MXN=X']
        df_MXN['IEMB.MI'] = df['IEMB.MI']*df['EURMXN=X']
        df_MXN['STIP'] = df['STIP']*df['MXN=X']
        df_MXN['IAU'] = df['IAU']*df['MXN=X']
        df_MXN['EWZ'] = df['EWZ']*df['MXN=X']
        df_MXN = df_MXN.dropna()
        df_MXNN = df_MXN.pct_change().dropna()

        #Información del sp500
        SP = download_datas('^GSPC','2021-01-01',None)
        SP = pd.DataFrame(SP)
        SP.columns=['S&P']

        portafolio = st.selectbox('Selecciona el tipo de portafolio', ['Portafolio optimizado por Sharpe Ratio', 'Portafolio de minima varianza', 'Portafolio Equally Weighted',  'Minima volatilidad con rendimiento del 10%', 'Índice S&P500','Conclusiones'])

        if portafolio == 'Portafolio optimizado por Sharpe Ratio':
            ####
            st.header("1. Portafolio optimizado por Sharpe Ratio")

            st.subheader("Estadísticas del portafolio Sharpe Ratio.")
            weight_d1=[0.638,0.1188,0.0,0.2432,0.0]
            inciso_d1,ACd1=calculos_est(weight_d1,df_MXN,0)
            st.write(inciso_d1)

            st.subheader("Rendimientos acumulados")
            st.write(ACd1)

            #rendimientos
            #df_MXNN = df_MXN.pct_change().dropna()
            df_pesos1 = df_MXNN*weight_d1
            retornos1 = pd.DataFrame(df_pesos1.sum(axis=1))

            st.header("Drawdown")

            #Se refiere a cuánto ha caído una cuenta desde su punto máximo hasta su punto mínimo.
            # Calcular la serie acumulativa de rendimientos
            df_acumulativo = (1+retornos1).cumprod()

            # Calcular el pico acumulativo previo en cada punto del tiempo
            df_pico = df_acumulativo.cummax()

            # Calcular la serie de drawdowns
            df_drawdown = (df_acumulativo - df_pico)/df_pico

            # Encontrar el drawdown máximo
            df_max_drawdown = df_drawdown.min()
            df_max_drawdown.columns = ['%']

            st.markdown(f"En su peor momento, el portafolio experimentó la siguiente caída en porcentaje respecto su pico anterior")
            st.write(df_max_drawdown)

            st.markdown(f"Los últimos valores de la serie de drawdowns del portafolio es:")
            st.write(df_drawdown.tail())

        elif portafolio == 'Portafolio de minima varianza':
            ####
            st.header("2. Portafolio de minima varianza")

            st.subheader("Estadísticas del portafolio.")
            weight_d2=[0.444,0.3309,0.5429,0.0295,0.0523]
            inciso_d2,ACd2=calculos_est(weight_d2,df_MXN,0)
            st.write(inciso_d2)
            st.subheader("Rendimientos acumulados")
            st.write(ACd2)

            #rendimientos
            df_pesos2 = df_MXNN*weight_d2
            retornos2 = pd.DataFrame(df_pesos2.sum(axis=1))

            st.header("Drawdown")

            #Se refiere a cuánto ha caído una cuenta desde su punto máximo hasta su punto mínimo.
            # Calcular la serie acumulativa de rendimientos
            df_acumulativo = (1+retornos2).cumprod()

            # Calcular el pico acumulativo previo en cada punto del tiempo
            df_pico = df_acumulativo.cummax()

            # Calcular la serie de drawdowns
            df_drawdown = (df_acumulativo - df_pico)/df_pico

            # Encontrar el drawdown máximo
            df_max_drawdown = df_drawdown.min()
            df_max_drawdown.columns = ['%']

            st.markdown(f"En su peor momento, el portafolio experimentó la siguiente caída en porcentaje respecto su pico anterior")
            st.write(df_max_drawdown)

            st.markdown(f"Los últimos valores de la serie de drawdowns del portafolio es:")
            st.write(df_drawdown.tail())


        elif portafolio == 'Portafolio Equally Weighted':
            ###
            st.header("3. Portafolio Equally Weighted")

            st.subheader("Estadísticas del portafolio.")
            weight_d3=[0.2,0.2,0.2,0.2,0.2]
            inciso_d3,ACd3=calculos_est(weight_d3,df_MXN,0)
            st.write(inciso_d3)

            st.subheader("Rendimientos acumulados")
            st.write(ACd3)

            #rendimientos
            df_pesos3 = df_MXNN*weight_d3
            retornos3 = pd.DataFrame(df_pesos3.sum(axis=1))

            st.header("Drawdown")

            #Se refiere a cuánto ha caído una cuenta desde su punto máximo hasta su punto mínimo.
            # Calcular la serie acumulativa de rendimientos
            df_acumulativo = (1+retornos3).cumprod()

            # Calcular el pico acumulativo previo en cada punto del tiempo
            df_pico = df_acumulativo.cummax()

            # Calcular la serie de drawdowns
            df_drawdown = (df_acumulativo - df_pico)/df_pico

            # Encontrar el drawdown máximo
            df_max_drawdown = df_drawdown.min()
            df_max_drawdown.columns = ['%']

            st.markdown(f"En su peor momento, el portafolio experimentó la siguiente caída en porcentaje respecto su pico anterior")
            st.write(df_max_drawdown)

            st.markdown(f"Los últimos valores de la serie de drawdowns del portafolio es:")
            st.write(df_drawdown.tail())

        elif portafolio == 'Minima volatilidad con rendimiento del 10%':
            ###
            st.header("4. Minima volatilidad con rendimiento del 10%")

            st.subheader("Estadísticas del portafolio.")
            weight_d4=[0.3649,0.2691,0.2006,0.1654,0.00]
            inciso_d4,ACd4=calculos_est(weight_d4,df_MXN,0)
            st.write(inciso_d4)

            st.subheader("Rendimientos acumulados")
            st.write(ACd4)

            #rendimientos
            df_pesos4 = df_MXNN*weight_d4
            retornos4 = pd.DataFrame(df_pesos4.sum(axis=1))

            st.header("Drawdown")

            #Se refiere a cuánto ha caído una cuenta desde su punto máximo hasta su punto mínimo.
            # Calcular la serie acumulativa de rendimientos
            df_acumulativo = (1+retornos4).cumprod()

            # Calcular el pico acumulativo previo en cada punto del tiempo
            df_pico = df_acumulativo.cummax()

            # Calcular la serie de drawdowns
            df_drawdown = (df_acumulativo - df_pico)/df_pico

            # Encontrar el drawdown máximo
            df_max_drawdown = df_drawdown.min()
            df_max_drawdown.columns = ['%']

            st.markdown(f"En su peor momento, el portafolio experimentó la siguiente caída en porcentaje respecto su pico anterior")
            st.write(df_max_drawdown)

            st.markdown(f"Los últimos valores de la serie de drawdowns del portafolio es:")
            st.write(df_drawdown.tail())


        elif portafolio == 'Índice S&P500':
        ###
            st.header("Índice S&P500")

            st.subheader("Estadísticas del índice.")
            weight_SP=[1]
            inciso_d_SP,ACSP=calculos_est(weight_SP,SP,1)
            st.write(inciso_d_SP)

            st.subheader("Rendimientos acumulados")
            st.write(ACSP)

            #rendimientos
            df_pesos5 = SP*weight_SP
            retornos5 = pd.DataFrame(df_pesos5.sum(axis=1))

            st.header("Drawdown")

            #Se refiere a cuánto ha caído una cuenta desde su punto máximo hasta su punto mínimo.
            # Calcular la serie acumulativa de rendimientos
            df_acumulativo = (1+retornos5).cumprod()

            # Calcular el pico acumulativo previo en cada punto del tiempo
            df_pico = df_acumulativo.cummax()

            # Calcular la serie de drawdowns
            df_drawdown = (df_acumulativo - df_pico)/df_pico

            # Encontrar el drawdown máximo
            df_max_drawdown = df_drawdown.min()
            df_max_drawdown.columns = ['%']

            st.markdown(f"En su peor momento, el portafolio experimentó la siguiente caída en porcentaje respecto su pico anterior")
            st.write(df_max_drawdown)

            st.markdown(f"Los últimos valores de la serie de drawdowns del portafolio es:")
            st.write(df_drawdown.tail())

        elif portafolio == 'Conclusiones':
######
            st.header("Conclusiones")

            weight_d1=[0.638,0.1188,0.0,0.2432,0.0]
            inciso_d1,ACd1=calculos_est(weight_d1,df_MXN,0)

            weight_d2=[0.444,0.3309,0.5429,0.0295,0.0523]
            inciso_d2,ACd2=calculos_est(weight_d2,df_MXN,0)

            weight_d3=[0.2,0.2,0.2,0.2,0.2]
            inciso_d3,ACd3=calculos_est(weight_d3,df_MXN,0)

            weight_d4=[0.3649,0.2691,0.2006,0.1654,0.00]
            inciso_d4,ACd4=calculos_est(weight_d4,df_MXN,0)

            weight_SP=[1]
            inciso_d_SP,ACSP=calculos_est(weight_SP,SP,1)

            final=pd.concat([inciso_d1,inciso_d2,inciso_d3,inciso_d4, inciso_d_SP])
            indices_f=['SR','MV','EW','MV10%' ,'S&P']
            final.reset_index(inplace=True)
            final=final.rename(columns={'Ticker':'Portafolio'})
            final['Portafolio']=indices_f
            final.set_index('Portafolio',inplace=True)
            st.write(final)
 #########################################################################################################################################################################
            
            st.markdown("Podemos observar un dataframe en donde se exponen los resultados de las estadísticas a partir de obtener los ponderados respecto a las optimizaciones de los portafolios SharpRatio(SR), Mínima Volatilidad (MN), Equally Weighted(EW), Mínima Varianza con rendimiento del 10%(MV10%) y del activo S&P 500 (S&P).")
            st.markdown("De acuerdo a los resultados de las estadísticas llegamos a las siguientes conclusiones:")
            st.markdown("Para el caso del portafolio SR:")
            st.markdown("Se observa que para el caso de la media es positiva, pero se encuentra por debajo de la media del S&P. De igual forma cuenta con una desviación estándar baja a comparación de los demás portafolios, dándonos a entender que se trata de un portafolio de bajo riesgo.")
            st.markdown("Para el caso del VaR y el CVaR son menores al del S&P. Finalmente se puede observar que para el caso del Sharpe Ratio y Sortino son menores a las del S&P, por lo que nos da a entender que el rendimiento ajustado al riesgo es menor al del S&P.")
            st.markdown("Para el caso del portafolio MV:")
            st.markdown("Para el caso de la media se encuentra por debajo del cero y su desviación estándar son bastante similares, dándonos a entender que tiene el mismo riesgo que el S&P, pero de igual forma hay que considerar que el VaR y el CVaR son mayores al del S&P, por lo que indica que para el caso de las métricas de riesgo hay un mayor riesgo a pesar de tener una desviación estándar similar.")
            st.markdown("Para el caso de SharpeRatio y Sortino es análogo al portafolio anterior teniendo un rendimiento ajusto al riesgo menor al S&P.")
            st.markdown("Para el portafolio EW:")
            st.markdown("Podemos observar que al igual que MV, la media del portafolio EW se encuentra por debajo del cero y además la desviación estándar se encuentra ligeramente por debajo del S&P, lo cual nos indica que el portafolio es menos riesgoso que el S&P.")
            st.markdown("Para el caso del VaR y el CVaR nos afirman igual que el riesgo es menor ya que las métricas se encuentran por debajo de las del S&P. Finalmente, para el Sharpe Ratio y el Sortino nos indica que el rendimiento ajusto al riesgo es menor al S&P.")
            st.markdown("Portafolio MV & R=10%:")
            st.markdown("Tiene una media ligeramente negativa pero mayor a la del portafolio de mínima varianza.")
            st.markdown("Sharpe Ratio y Sortino Ratio son negativos, con una volatilidad más alta ante las caídas de precio.")
            st.markdown("Sus 3 varianzas menores que las del S&P.")
            st.markdown("En conclusión, consideramos que el portafolio más óptimo es el de mínima varianza con un rendimiento fijo de 10%, dado que es el portafolio que tiene un mayor rendimiento en comparación con el de mínima varianza y el de Equally Weighted con una desviación estándar también menor. Es decir, de acuerdo a nuestro análisis este portafolio es el que tiene mejor relación proporcional entre el su volatilidad y el rendimiento esperado (promedio).")
            st.markdown("Por otro lado, la diferencia entre el Sharpe ratio y Sortino es menor que la de los otros portafolios. Y esto nos dice, que en comparación con las otras optimizaciones, este portafolio tiene una menor volatilidad nate las bajas.")
##################################################################################################################################################################
###################



    if portafolio == "B":

        st.title("Portafolios óptimos")
        symbols = ['IVV', 'IEMB.MI', 'STIP', 'IAU', 'EWZ', 'MXN=X', 'EURMXN=X']

        # Numero de activos de nuestro portafolio
        symbols2 = ['IVV', 'IEMB.MI', 'STIP', 'IAU', 'EWZ']
        numofasset = len(symbols2)

        numofportfolio = 100000

        #Descargamos datos
        df=download_datas(symbols,'2010-01-01',"2020-12-31")

        #Pasamos todos los retornos a MXN
        df_MXN = pd.DataFrame(columns= ['IVV', 'IEMB.MI', 'STIP', 'IAU', 'EWZ'])
        df_MXN['IVV'] = df['IVV']*df['MXN=X']
        df_MXN['IEMB.MI'] = df['IEMB.MI']*df['EURMXN=X']
        df_MXN['STIP'] = df['STIP']*df['MXN=X']
        df_MXN['IAU'] = df['IAU']*df['MXN=X']
        df_MXN['EWZ'] = df['EWZ']*df['MXN=X']

        # Calculamos retornos
        returns = df_MXN.pct_change().fillna(0)

            #Establecer los rangos de 0 a 1
        tuple((0, 1) for x in range(numofasset))

        # Especificar restricciones y limites
        cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
        bnds = tuple((0, 1) for x in range(numofasset))
        initial_wts = numofasset*[1./numofasset]

 #########
        optimo = st.selectbox('Selecciona el tipo de portafolio', ['Portafolio optimizado por Sharpe Ratio', 'Portafolio de minima varianza', 'Minima volatilidad con rendimiento del 10%'])

        if optimo == 'Portafolio optimizado por Sharpe Ratio':

            st.header('Portafolio de Max Sharpe Ratio')

            # Optimizacion por maximo Shape ratio
            opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
            #opt_sharpe

            st.subheader("Pesos del portafolio")
            # Deplegamos los pesos del portafolio con Max Shape Ratio
            pesos1 = (list(zip(symbols2, around(opt_sharpe['x']*100,2))))

            # Convertir la lista en un DataFrame de Pandas
            pesotes1 = pd.DataFrame(pesos1, columns=['ETF (Ticker)', 'Peso'])
            st.write(pesotes1)

            st.subheader("Estadisticas")
            # Estadisticas del portafolio
            stats = ['Returns', 'Volatility', 'Sharpe Ratio']
            est1 = list(zip(stats, around(portfolio_stats(opt_sharpe['x']),4)))
            estadistica1 = pd.DataFrame(est1, columns=['Estadistica', 'Valor'])
            st.write(estadistica1)

        if optimo == 'Portafolio de minima varianza':
            st.header('Portafolio de Mínima Varianza')
            #Optimizacion de minima varianza
            opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

            st.subheader("Pesos del portafolio")
            # Pesos de portafolio de minima varianza
            pesos2 = (list(zip(symbols2, around(opt_var['x']*100,2))))

            # Convertir la lista en un DataFrame de Pandas
            pesotes2 = pd.DataFrame(pesos2, columns=['ETF (Ticker)', 'Peso'])
            st.write(pesotes2)

            # Métricas del portafolio
            st.subheader("Estadisticas")
            stats = ['Returns', 'Volatility', 'Sharpe Ratio']
            est2 = list(zip(stats, around(portfolio_stats(opt_var['x']),4)))
            estadistica2 = pd.DataFrame(est2, columns=['Estadistica', 'Valor'])
            st.write(estadistica2)

        if optimo == 'Minima volatilidad con rendimiento del 10%':
            st.header('Portafolio de minima volatilidad con rendimiento del 10%')
            targetrets = linspace(0.10,0.60,100)
            tvols = []

            for tr in targetrets:
                ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
                          {'type': 'eq', 'fun': lambda x: sum(x) - 1})
                opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)
                tvols.append(opt_ef['fun'])
            targetvols = array(tvols)

           # Data frame para la forntera eficiente
            st.subheader("Frontera eficiente")
            efport = pd.DataFrame({'targetrets' : around(100*targetrets,2),'targetvols': around(100*targetvols,2),'targetsharpe': around(targetrets/targetvols,2)})
            st.write(efport)

            # Crear la figura de Plotly
            st.subheader("Gráfico frontera eficiente")
            fig_size = (800, 600)

            fig = px.scatter(
                efport, x='targetvols', y='targetrets',  color='targetsharpe',
                labels={'targetrets': 'Expected Return', 'targetvols': 'Expected Volatility','targetsharpe': 'Sharpe Ratio'},
                title="Efficient Frontier Portfolio"
            ).update_traces(mode='markers', marker=dict(symbol='cross'))

            fig.update_xaxes(showspikes=True)
            fig.update_yaxes(showspikes=True)

            fig.update_layout(
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                width=fig_size[0],
                height=fig_size[1]
            )

            st.plotly_chart(fig)


            #Definimos restriciones
            cons2 = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},  {'type': 'eq', 'fun': lambda x: returns_means(x) - 0.1} )
            opt_var10 = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=cons2)
            pesos3 = list(zip(symbols2, around(opt_var10['x']*100,2)))

            st.subheader("Pesos del portafolio")
            # Convertir la lista en un DataFrame de Pandas
            pesotes3 = pd.DataFrame(pesos3, columns=['ETF (Ticker)', 'Peso'])
            st.write(pesotes3)

            st.subheader("Estadisticas")
            # Estadisticas del portafolio
            stats = ['Returns', 'Volatility', 'Sharpe Ratio']
            est3 = list(zip(stats, around(portfolio_stats(opt_var10['x']),4)))
            estadistica3 = pd.DataFrame(est3, columns=['Estadistica', 'Valor'])
            st.write(estadistica3)


#
#
#


if option == "Activos":
    st.title("Resumen y Estadisticas del activo")
    activo = st.sidebar.selectbox(
        "Elige un activo",
        ('IEMB.MI', 'EWZ', "STIP", "IVV", "IAU")
    )
    df_activo = df_activos[activo]

    st.header("Precios")
    # Crear la figura
    fig = go.Figure()

    # Agregar los datos del activo a la figura
    fig.add_trace(go.Scatter(x=df_activo.index, y=df_activo.values, mode='lines'))

    # Establecer títulos y etiquetas
    fig.update_layout(title='Precio de cierre historico del activo',
                    xaxis_title='Fecha',
                    yaxis_title='Precio de Cierre (en $)')

    st.plotly_chart(fig)

    # Rendimientos
    st.header("Rendimientos")
    st.markdown("Elige la fecha a la que quieres los rendimientos cuidando que no sea un fin de semana o el día actual:")

    hoy = st.date_input('Introduce la fecha')
    hoy = pd.Timestamp(hoy)
    ventanas_de_tiempo = calcular_fechas(hoy)

    df_rendimientos = calcular_rendimiento(activos, ventanas_de_tiempo, activo)

    st.dataframe(df_rendimientos)


# Información por activo

if option == "Activos":
    if activo == "IEMB.MI":
        st.header("Información general")
        st.markdown("El ETF **IEMB** pretende replicar la rentabilidad de un índice compuesto por bonos denominados en dólares estadounidenses de países emergentes, por lo que al tenerlo en nuestro portafolio, nos encontraremos expuestos a diversos bonos guberamentales y cuasi-gubernamentales de mercados emergentes emitidos en dólares, los cuales tienen un grado de grado de inversión y de alto rendimiento.")
        st.markdown("El fondo se constituyo el 15 de Febrero del 2008 y cuenta con 57,372,365 activos de circulación al 01 de Diciembre del 2023")
        st.markdown("Su índice de referencia es JPMorgan EMBI Global Core Index (JPEICORE), el cual esta compuesto por bonos gubernamentales denominados en dólares estadounidenses emitidos por países de mercados emergentes, contando con una beta de 1.00 al 30/Nov/2023, indicando que nuestro ETF y su indice registran un mismo comportamiento.")

        st.header("Posiciones principales")
        st.write("Nuestro ETF tiene un total de 615 posiciones, todas con Renta Fija como Clase de activo, de entre las cuales 10 principales son las siguientes;")
        emisor = ["TURKEY (REPUBLIC OF)","SAUDI ARABIA (KINGDOM OF)","BRAZIL FEDERATIVE REPUBLIC OF (GOVERNMENT)",
         "PHILIPPINES (REPUBLIC OF)", "COLOMBIA (REPUBLIC OF)", "DOMINICAN REPUBLIC (GOVERNMENT)", "MEXICO (UNITED MEXICAN STATES) (GOVERNMENT)",
          "QATAR (STATE OF)", "INDONESIA (REPUBLIC OF)", "OMAN SULTANATE OF (GOVERNMENT)"	]
        peso = [4.48, 3.93, 3.80, 3.54, 3.41, 3.39, 3.27, 3.25, 3.04, 2.97]
        Tabla_posiciones = pd.DataFrame(list(zip(emisor, peso)), columns=['Emisor', 'Peso (%)'])
        st.write(Tabla_posiciones)

        st.header(" Exposición")
        exposicion = st.selectbox('Selecciona el tipo de exposición', ['Geográfica', 'Vencimiento', 'Cálidad crediticia'])

        if exposicion == 'Geográfica':
            st.subheader("Geográfica")
            st.markdown("Dentro de este ETF se puede apreciar una gran diversificación de los paises a los que tiene exposición, dado que si bien hay algunos con un poco mas de porcentaje, no hay mucha diferencia entre sí, entro los porcentajes más altos, encontramos a México, Arabia Saudita, Turquía e Indonesia, con un porcentaje mayor al 5% cada una como se muestra a continuación.")

            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            paises_destacados = {"Mexico": 5.91,"Saudi Arabia": 5.73,"Turkey": 5.24,"Indonesia": 5.19,"United Arab Emirates": 4.60,"Qatar": 4.11,"China": 3.81,"Brazil": 3.80,
                                 "Oman": 3.64,"Philippines": 3.61,"Colombia": 3.41,"Dominican Republic": 3.39,"Chile": 3.35,"South Africa": 3.13,"Peru": 2.94,"Panama": 2.89,"Bahrain": 2.75,"Hungary": 2.59,
                                 "Egypt": 2.46,"Uruguay": 2.22,"Romania": 2.17,"Poland": 2.04,"Malaysia": 2.03,"Nigeria": 2.01,"Argentina": 1.80,"Angola": 1.15,"Costa Rica": 1.10,"Liquidity": 0.21,"Other": 12.70}
            world['Destacado'] = world['name'].map(lambda x: paises_destacados.get(x, 0))
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            world.boundary.plot(ax=ax, linewidth=1)
            world.plot(column='Destacado', cmap='OrRd', ax=ax, legend=True, legend_kwds={'label': "Porcentaje destacado (%)", 'orientation': "horizontal"})
            #st.pyplot(fig)

            paises = ["México", "Arabia Saudita", "Turquía", "Indonesia", "Emiratos Árabes Unidos", "Qatar", "China", "Brasil", "Omán", "Filipinas", "Colombia", "República Dominicana",
            "Chile", "Sudáfrica", "Perú", "Panamá", "Baréin", "Hungría", "Egipto", "Uruguay", "Rumania", "Polonia", "Malasia", "Nigeria", "Argentina", "Angola", "Costa Rica", "Liquidez", "Otros"]
            porcentaje = [5.91, 5.73, 5.24, 5.19, 4.60, 4.11, 3.81, 3.80, 3.64, 3.61, 3.41, 3.39, 3.35, 3.13, 2.94, 2.89, 2.75, 2.59, 2.46, 2.22, 2.17, 2.04, 2.03, 2.01, 1.80, 1.15, 1.10, 0.21, 12.70]
            Tabla_paises = pd.DataFrame(list(zip(paises, porcentaje)), columns=['Pais', 'Porcentaje'])
            #st.write(Tabla_paises)

            col1, col2 = st.columns([2, 4])
            col1.write(Tabla_paises)
            col2.pyplot(fig)

        elif exposicion == 'Vencimiento':
            st.subheader("Vencimiento")
            st.markdown("En su mayoría vemos una alta exposición a bonos de largo plazo con poco mas de una cuarta parte de dicho portafolio invertido en un plazo a más de 20 años indicando muy poca liquidez")
            vencimiento = ['Efectivo y derivados', '0 a 1 año', '1 a 2 años', '2 a 3 años', '3 a 5 años', '5 a 7 años', '7 a 10 años', '10 a 15 años', '15 a 20 años', 'Más de 20 años']
            porcentajes = [0.25, 2.15, 6.03, 6.93, 16.17, 13.00, 18.13, 6.66, 4.26, 26.44]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(vencimiento, porcentajes, color='cornflowerblue')
            ax.set_xlabel('Porcentaje (%)')
            ax.set_title('Distribución de Activos por Plazo de Inversión')
            st.pyplot(fig)

        elif exposicion == 'Cálidad crediticia':
            st.subheader('Cálidad crediticia')
            st.markdown("La cálidad crediticia de un poco más de la mitad de este portafolio tiene una calificación entre AA, A y BBB, se debe tener cuidado puesto que la otra parte del portafolio se encuentra por debajo del grado de inversión, entrando en un grado especulativo ")
            categorias = ['Liquidez', 'Calificación AA', 'Calificación A', 'Calificación BBB', 'Calificación BB', 'Calificación B', 'Calificación CCC', 'Calificación CC', 'Calificación C', 'Calificación D', 'Sin calificación']
            porcentajes_calificaciones = [0.25, 7.52, 16.62, 27.99, 21.49, 19.08, 2.42, 2.53, 0.00, 1.76, 0.35]
            # Crear el gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            k = ax.bar(categorias, porcentajes_calificaciones, color='cornflowerblue')
            ax.set_xlabel('Categoría')
            ax.set_ylabel('Porcentaje (%)')
            ax.set_title('Distribución de Activos por Calificación')
            ax.tick_params(axis='x', rotation=45)
            for i in k:
                yval = i.get_height()
                plt.text(i.get_x() + i.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8, color='black')
            st.pyplot(fig)


    elif activo == "EWZ":
        st.header("Información general")
        st.markdown("El ETF **EWZ** replica los resultados de inversión de un índice compuesto por valores de renta variable de Brasil, dando exposición a grandes y medianas empresas con un acceso del  85 % del mercado de acciones brasileñas. El fondo fue constituido el 10 de Julio del 2000, con USD como su divisa base")
        st.markdown("Su índice de referencia es MSCI Brazil 25/50 Index(M1BR2550), cuenta con una beta de 0.99 indicando que el ETF tiende a ser ligeramente menos volátil que el mercado dado su índice de referencia")

        st.header("Posiciones principales")
        st.markdown("Observacion. En su mayoria las clases de activos son equity")
        data = {
          'Ticker': ['VALE3', 'PETR4', 'ITUB4', 'PETR3', 'BBDC4', 'B3SA3', 'ABEV3', 'WEGE3', 'RENT3', 'BBAS3'],
          'Nombre': ['CIA VALE DO RIO DOCE SH', 'PETROLEO BRASILEIRO PREF SA', 'ITAU UNIBANCO HOLDING PREF SA', 'PETROLEO BRASILEIRO SA PETROBRAS', 'BANCO BRADESCO PREF SA', 'B3 BRASIL BOLSA BALCAO SA', 'AMBEV SA', 'WEG SA', 'LOCALIZA RENT A CAR SA', 'BANCO DO BRASIL SA'],
          'Sector': ['Materiales', 'Energía', 'Financieros', 'Energía', 'Financieros', 'Financieros', 'Productos básicos de consumo', 'Industriales', 'Industriales', 'Financieros'],
          'Clase de activos': ['Equity']*10,
          'Peso (%)': [12.75, 8.37, 7.69, 6.82, 4.22, 3.95, 3.27, 3.14, 3.0, 2.59]         }
        dfposiciones = pd.DataFrame(data)
        st.write(dfposiciones)

        st.header("Exposición")
        st.markdown("Se puede observar un portafolio diversificado sectorialmente, tenemos alrededor de un 25% de nuestro portafolio invertido en acciones financieras, aproximadamente 40% en el sector de energía y materiales,  al rededor de 25% entre servicios, industriales y básicos de consumo y el demás porcentaje dividido en otros sectores.")
        sectores = ['Financieros', 'Energía', 'Materiales', 'Servicios', 'Industriales','Productos básicos de consumo', 'Cuidado de la Salud', 'Consumo discrecional','Comunicación', 'Efectivo y Derivados', 'Tecnología de la Información']
        porcentajes = [25.89, 19.14, 17.92, 9.75, 8.84, 8.05, 2.55, 2.44, 2.41, 2.24, 0.76]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sectores,porcentajes, color='cornflowerblue')
        ax.set_title('Distribución de Sectores en (%)')
        st.pyplot(fig)


    elif activo == "STIP":
        st.header("Información general")
        st.markdown("EL ETF **STIP** busca replicar el índice ICE US Treasury 0-5 Year Inflation Linked Bond Index (USD) (CETIPO), el cual se encuentra compuesto por bonos del Tesoro de EE.UU. protegidos contra la inflación y amortizaciones menores a 5 años, el ETF cuenta con una beta de 0.12 lo que nos indica que es mucho menos volatil y por ende menos riesgoso que el indice que esta replicando")
        st.markdown("Fue creado el 01 de Diciembre del 2010 y tiene USD como divisa base. Este se encuentra expuesto a U.S. TIPS, bonos del gobierno cuyo valor nominal aumenta con la inflación, teniendo un acceso dirigido a una sección especifica del mercado doméstico de TIPS")

        st.header("Exposición")
        st.markdown("Dada la composición que tiene, encontramos que su mayor emisor es UNITED STATES TREASURY en un 99.99%, por lo  mismo se encuentra enfocada su inversion en Bonos del Tesoro en un 99.99% y el 0.01% restante se encuentra en derivados y liquidez, por ende su cálidad crediticia es de alto grado, pues la calificación corresponde a un AA.")
        st.markdown("Encontramos entonces que este ETF no tiene mucha diversificación a excepción del vencimiento de sus bonos que va desde los 0-5 años dado el índice que busca replicar, teniendo aproximadamente un 40% de su portafolio en bonos a vencer en 3-5 años y un 20% a un año por lo que no posee un alto riesgo de liquidez.")

        vencimiento = ['0 a 1 año', '1 a 2 años', '2 a 3 años', '3 a 5 años']
        porcentajes = [19.91, 22.05, 16.77, 41.27]
        fig, ax = plt.subplots()
        ax.pie(porcentajes, labels=vencimiento, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'royalblue', 'steelblue', 'cornflowerblue'])
        ax.axis('equal')  # Equal aspect ratio asegura que el pastel sea circular.
        plt.title("Vencimiento de los bonos")
        st.pyplot(fig)


    elif activo == "IVV":
        st.header("Información general")
        st.markdown("El ETF **IVV**, replica al conocido índice S&P 500 (SPTR) compuesto de renta variable de alta capitalización de EE.UU, de ese modo este ETF tiene exposición a 500 grandes empresas establecidas en EE.UU, ofreciendo además un bajo costo por dicha exposición. Su respectiva Beta es de 1.00 por lo que hay un mismo comportamiento entre el ETF y el índice")

        st.header("Posiciones principales")
        st.markdown("Podemos observar que en sus principales posiciones tenemos empresas tipo growth, ya que principalmente sus acciones pertenecen al sector de tecnología.")
        st.markdown("Observación. En su mayoría las clases de activos son equity")
        data = {'Ticker': ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRKB', 'UNH'],
                'Nombre': ['APPLE INC', 'MICROSOFT CORP', 'AMAZON COM INC', 'NVIDIA CORP', 'ALPHABET INC CLASS A', 'META PLATFORMS INC CLASS A', 'ALPHABET INC CLASS C', 'TESLA INC', 'BERKSHIRE HATHAWAY INC CLASS B', 'UNITEDHEALTH GROUP INC'],
                 'Sector': ['Tecnología de la Información', 'Tecnología de la Información', 'Consumo discrecional', 'Tecnología de la Información', 'Comunicación', 'Comunicación', 'Comunicación', 'Consumo discrecional', 'Financieros', 'Cuidado de la Salud'],
                 'Clase de activo': ['Equity'] * 10,
                 'Peso (%)': [7.24, 7.13, 3.42, 2.92, 1.99, 1.85, 1.71, 1.69, 1.69, 1.32] }
        tabla_posicion = pd.DataFrame(data)
        st.table(tabla_posicion)

        st.header("Exposición")
        st.markdown("Hay una exposición sectorial muy diversificada, donde el mayor peso del portafolio cae en acciones del sector tecnología de la información en aproximadamente un 30%.")
        sector = ['Tecnología de la Información', 'Financieros', 'Cuidado de la Salud', 'Consumo discrecional', 'Comunicación', 'Industriales', 'Productos básicos de consumo', 'Energía', 'Inmobiliario', 'Materiales', 'Servicios', 'Efectivo y Derivados']
        porcentajes = [28.63, 13.00, 12.71, 10.77, 8.44, 8.40, 6.31, 4.09, 2.49, 2.43, 2.41, 0.32]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sector, porcentajes, color='cornflowerblue')
        ax.set_xlabel('(%)')
        ax.set_title('Distribución Sectorial')
        st.pyplot(fig)

    elif activo == "IAU":
        st.header("Información general")
        st.markdown("El ETF **IAU** busca replicar al LBMA Gold Price (LBMA Gold Price). Este ETF tiene una beta de 0.08, por lo que su volatilidad y su riesgo con el mercado en comparación al índice es un poco menor.")
        st.markdown("Este ETF dado el índice que busca replicar da una exposición a los movimientos diarios del lingote de oro, dando un cómodo acceso al precio del lingote de oro. Además se encuentra respaldado por oro físico, contando con 396,10 toneladas en custodia al 05 de Diciembre del 2023")
        st.markdown("Fue constituido el 25 de Enero del 2005, tiene a USD como divisa base y su clase de activo dada su composición es materias primas")
