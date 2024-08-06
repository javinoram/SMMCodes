from base import hamiltonian, valor_esperado
from base import J3D, J1D, OZ
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

Phase = []
for hz in Hz:
    #Construccion del hamiltoniano
    H = hamiltonian([j1, j2, j3, j, hz, hx, int(conf)])

    #Calculo de valores y vectores propios
    ee, vv= np.linalg.eigh(H)

    #Calculo de valores esperados del S_z respecto a cada vector propio
    proy = np.array( [ ((vv[:,k]).T.conj()).dot(OZ).dot(vv[:,k]) for k in range(len(ee))] )

    #Calculo de magnetizacion
    valuesbase = [valor_esperado(ee, proy, t, 120) for t in T]
    Phase.append( valuesbase )

#Almacenar resultados en un .csv  
Phase = pd.DataFrame( Phase )
Phase.to_csv("datos/two-mol/mag/"+structure+conf+"j"+str(j)+"hx"+str( hx )+".csv")
