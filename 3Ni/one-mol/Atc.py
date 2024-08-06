import numpy as np
import pandas as pd
from scipy import integrate
from scipy import interpolate
import sys
from base import hamiltoniano, specific_heat, valor_esperado
from base import J3D, J1D, OZ, nub, gyro, dtype

## Mapas de fase que seran usados para la interpolacion
def Total_map_mag(M, temp, rango_integracion, params):
    Phase_map = []
    for h in rango_integracion:
        params[3] = h
        ee, vv = np.linalg.eigh( hamiltoniano([params[0], params[1], params[2], params[3], params[4]]) )
        proy = np.array([ (vv[:,v]).conj().T.dot(M).dot(vv[:,v]) for v in range(len(ee)) ])
        proy = np.round( np.real( proy ) ,7)
        tmp_list = [valor_esperado(ee, proy, t, 150) for t in temp]
        Phase_map.append( tmp_list )
    return Phase_map

def Total_map_sh(temp, rango_integracion, params):
    Phase_map = []
    for h in rango_integracion:
        params[3] = h
        ee = np.linalg.eigvalsh( hamiltoniano([params[0], params[1], params[2], params[3], params[4]]) )
        tmp_list = [specific_heat(ee, None, t, 150) for t in temp]
        Phase_map.append( tmp_list )
    return Phase_map


def auxiliar_fun(f, g, t):
    np.seterr(all='raise')
    derivada_mag = f(t)
    try:
        valor = np.divide( derivada_mag, g(t), dtype=dtype)
    except FloatingPointError:
        if abs( np.round(derivada_mag, 10) ) < 1e-8:
            valor = 0.0
        else:
            valor = np.divide( np.round(derivada_mag, 10), g(t), dtype=dtype)
    return -nub*gyro*valor

def map_derivate_calculation(curvas_derivada_mag, curvas_calor_especifico, temp):
    map_derivate = []
    for f,g in zip(curvas_derivada_mag, curvas_calor_especifico):
        line = [ auxiliar_fun(f,g,t)  for t in temp ]
        map_derivate.append(line)
    return map_derivate

def integral(mapa_de_fase, rango_integracion, temperatura, upper_lim, dx):
    tmp_phase = mapa_de_fase
    lim_sup = len(rango_integracion[ rango_integracion<=upper_lim ])
    phase_retorno = [ temperatura[i]*integrate.trapezoid(line[:lim_sup], rango_integracion[:lim_sup], dx ) for i,line in enumerate(tmp_phase)]
    return phase_retorno


a = 0.001
b = 10.001
delta = 0.0025
numero = int( (b-a)/delta )
rango_integracion = np.array([np.round(a+ i*delta, 5) for i in range(numero)])
dx = np.round(delta, 5)

a = 0.5
b = 10.0
delta = 0.002
numero = int( (b-a)/delta )
temperatura_interpolar = np.array([np.round(a+ i*delta, 5) for i in range(numero)])


Op = OZ
structure = sys.argv[1]
hx = np.round( float(sys.argv[2]), 5 )
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D
params = [j1, j2, j3, 0.0, hx]

##fase 1, crear el mapa de fase del calor especifico y la magnetizacion
Phase_Map_mag = Total_map_mag(Op, temperatura_interpolar, rango_integracion, params)
Phase_Map_sh = Total_map_sh(temperatura_interpolar, rango_integracion, params)




##fase 2, interpolar las curvas
curvas_sh = []
for i,tmp in enumerate(Phase_Map_sh):
    f = interpolate.CubicSpline( temperatura_interpolar, tmp)
    curvas_sh.append( f )

curvas_mag = []
for i,tmp in enumerate(Phase_Map_mag):
    f = interpolate.CubicSpline( temperatura_interpolar, tmp)
    curvas_mag.append( f )

curvas_derivada_mag = []
for f in curvas_mag:
    curvas_derivada_mag.append( f.derivative(nu=1) )


## fase 3, usando las curvas interpoladas calcular los valores para la integral
a = 0.5
b = 10.0
delta = 0.002
numero = int( (b-a)/delta )
temperatura_calculo = np.array([np.round(a+ i*delta, 5) for i in range(numero)])
map_final = map_derivate_calculation(curvas_derivada_mag, curvas_sh, temperatura_calculo)
map_final = np.array(map_final).T
Phase_aux = pd.DataFrame( map_final )
Phase_aux.to_csv(f"datos/one-mol/itc/mapa_funcion_integrar{structure}hx{hx}.csv")



## fase 4, calcular la integral
Phase = []
hz = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0,
    5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0]
for h in hz:
    print(f"{h} campo {dx}")
    tmp_list = integral( map_final, rango_integracion, temperatura_calculo, h, dx )
    Phase.append(tmp_list)

Phase = pd.DataFrame( Phase )
Phase.to_csv(f"datos/one-mol/itc/{structure}hx{hx}.csv")
