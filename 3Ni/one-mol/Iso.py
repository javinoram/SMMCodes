from base import entropy, hamiltoniano
from base import J3D, J1D
import numpy as np
import pandas as pd
import sys

"""
Lectura de parametros de la linea de comandos, si es 3D y 1D.
"""
structure = sys.argv[1]
if structure == '3D':
    j1,j2,j3= J3D
else:
    j1,j2,j3= J1D

"""
Variables del sistema, campo magnetico en Z, X y la temperatura
"""
a = 0.001
b = 10.0
delta = 0.0015
numero = int( (b-a)/delta )
Hz = np.array([np.round(a+ i*delta, 5) for i in range(numero)])

Hx = [0.0, 0.5, 1.0]

a = 0.007
b = 9.1
delta = 0.0015
numero = int( (b-a)/delta )
T = np.array([np.round(a+ i*delta, 5) for i in range(numero)])

for hx in Hx:
    #Construccion del hamiltoniano
    H = hamiltoniano([j1, j2, j3, 0.0, hx])

    #Calculo de valores y vectores propios
    eeBase, vvBase= np.linalg.eigh(H)

    Phase = []
    for hz in Hz:
        #Construccion del hamiltoniano
        H = hamiltoniano([j1, j2, j3, hz, hx])

        #Calculo de valores y vectores propios
        ee, vv= np.linalg.eigh(H)

        #Calculo de la variacion de entropia (isothermal entropy change) respecto a campo hz 0 
        # y un campo hz variable, el campo hx es fijo durante todo el calculo.
        valuesbase = [entropy(eeBase, None, t, 120) - entropy(ee, None, t, 120) for t in T]
        Phase.append( valuesbase )

    #Almacenar resultados en un .csv
    Phase = pd.DataFrame( Phase )
    Phase.to_csv("datos/one-mol/iso/"+structure+"hx"+str( hx )+".csv")
