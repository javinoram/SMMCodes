from base import hamiltonian, specific_heat
from base import J3D, J1D
import sys
import numpy as np
import pandas as pd

"""
Lectura de parametros de la linea de comandos, si es 3D y 1D, conf es la variable
para determinar el tipo de acople.
"""
structure = sys.argv[1]
conf = sys.argv[2]
j = np.round( float(sys.argv[3]), 5 )
hx = np.round( float(sys.argv[4]), 5 )
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D

"""
Variables del sistema, campo magnetico en Z, X, la temperatura y
los valores de exchange entre moleculas
"""
#Hz = np.linspace(0.0, 10, 1201)
Hx = [0.0, 0.5, 1.0]#np.linspace(0.0, 10, 11)
#T = np.linspace(1e-2, 10, 1501)[4:]
exchanges = [-0.25, -0.125, -0.075, 0.075, 0.125, 0.25]

a = 0.001
b = 10.0
delta = 0.005
numero = int( (b-a)/delta )
Hz = np.array([np.round(a+ i*delta, 5) for i in range(numero)])

a = 0.007
b = 9.1
delta = 0.003
numero = int( (b-a)/delta )
T = np.array([np.round(a+ i*delta, 5) for i in range(numero)])

#for hx in Hx:
#      for j in exchanges:
Phase = []
for hz in Hz:
    #Construccion del hamiltoniano
    H = hamiltonian([j1, j2, j3, j, hz, hx, int(conf)])

    #Calculo de valores y vectores propios
    ee= np.linalg.eigvalsh(H)#Numpyget_eigen(H)

    #Calculo de calor especifico
    valuesbase = [specific_heat(ee, None, t, 120) for t in T]
    Phase.append( valuesbase )
                  
#Almacenar resultados en un .csv
Phase = pd.DataFrame( Phase )
Phase.to_csv("datos/two-mol/sh/"+structure+conf+"j"+str(j)+"hx"+str( hx )+".csv")
