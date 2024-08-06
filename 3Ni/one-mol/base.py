import numpy as np
import mpmath as mp

"""
Constantes utiles.
    dtype: tipo de las variables
    gyro: constante giroscopica
    nub: magneton de borh
    boltz: constante de boltzman
"""
dtype = 'float128'
boltz = 8.617333262e-2 #meV/K
gyro = 2.0
nub = 5.7883818066e-2 #meV/T

"""
Funcion para calcular la distribucion de probabilidades de data estado del sistema
input:
    ee (np.array(float)): valores de energia
    t (float): valor de temperatura
    pre (int): precision de los calculos al usar mpmath
output:
    array con las probabilidades de cada nivel de energia
"""
def prob_states(ee: np.array, t: float, pre: int) -> np.array:
    np.seterr(all='raise')
    beta = 1.0/(t*boltz)
    ee_var = -ee*beta
    try:
        partition = np.exp( ee_var, dtype=dtype )
        Z = np.sum(partition, dtype=dtype)
        partition = np.divide( partition, Z, dtype=dtype )
    except FloatingPointError:
        with mp.workdps(pre):
            #beta = mp.fdiv(1.0, (t*boltz) )
            partition = [ mp.exp( e ) for e in ee_var ]
            Z = mp.fdiv( 1.0, mp.fsum(partition) )
            partition = [ float( mp.fmul(p, Z) ) for p in partition ]
    return np.array(partition)


"""
Funcion para calcular el logaritmo de la funcion de particion
input:
    ee (array(float)): valores de energia
    t (float): valor de temperatura
    pre (int): precision de los calculos al usar mpmath
output:
    valor del logaritmo de la funcion de particion
"""
def log_z_function(ee: np.array, t: float, pre: int) -> np.array:
    np.seterr(all='raise')
    beta = 1.0/(t*boltz)
    ee_var = -ee*beta
    try:
        partition = np.exp( ee_var, dtype=dtype )
        Z = np.sum(partition, dtype=dtype)
        Z = np.log(Z)
    except FloatingPointError:
        with mp.workdps(pre):
            partition = [ mp.exp( e ) for e in ee_var ]
            Z = mp.fsum(partition)
            Z = mp.log(Z)
    return float(Z)


"""
Funcion para calcular la entropia a una temperatura fija
input:
    ee (array(float)): valores de energia
    proy(array(float)): valores de las proyecciones de los estados sobre un operador
    t (float): valor de temperatura
    pre (int): precision de los calculos al usar mpmath
output:
    valor de la entropia
"""
def entropy(ee: np.array, proy: np.array, t: float, pre: int) -> float:
    partition = np.round(np.array(prob_states(ee, t, pre)), 18)
    thermal = np.sum( partition*ee, dtype=dtype )
    free_energy = -boltz*log_z_function(ee, t, pre)
    return np.divide(thermal, t, dtype=dtype) - free_energy

"""
Funcion para calcular el calor especifico a una temperatura fija
input:
    ee (array(float)): valores de energia
    proy(array(float)): valores de las proyecciones de los estados sobre un operador
    t (float): valor de temperatura
    pre (int): precision de los calculos al usar mpmath
output:
    valor del calor especifico
"""
def specific_heat(ee: np.array, proy: np.array, t: float, pre: int) -> float:
    beta = np.divide( 1.0, (t*t*boltz) )
    partition = np.round(np.array(prob_states(ee, t, pre)), 18)

    entalpia = ( np.sum( partition*ee, dtype=dtype )**2 )
    entalpia_2 = np.sum( partition*(ee**2), dtype=dtype )
    tmp_var = (entalpia_2) - (entalpia)
    tmp_var = np.round(tmp_var, 11)
    return beta*tmp_var


"""
Funcion para calcular el valor esperado de un operador a una temperatura fija
input:
    ee (array(float)): valores de energia
    proy(array(float)): valores de las proyecciones de los estados sobre un operador
    t (float): valor de temperatura
    pre (int): precision de los calculos al usar mpmath
output:
    valor esperado
"""
def valor_esperado(ee: np.array, proy: np.array, t: float, pre: int) -> float:
    partition = np.round(np.array(prob_states(ee, t, pre)), 18)
    suma = np.sum( partition*proy, dtype=dtype )
    return suma



"""
Funcion para construir el hamiltoniano de la molecula 3Ni
input:
    params: lista con los valores de las constantes necesarias para construir el hamiltoniano.
        la lista se compone po J1, J2, J13, h_z, h_x.
output:
    H: matriz del hamiltoniano
""" 
def hamiltoniano(params):
    H = -2*params[0]*Int1 -2*params[1]*Int2 -2*params[2]*Int3 -gyro*nub*params[3]*OZ -gyro*nub*params[4]*OX
    H = np.real(H)
    return H


"""
Matrices de pauli de spin 1
""" 
sI = np.array( [ [1,0,0], [0,1,0], [0,0,1] ] ,dtype='float64')
sX = (1.0/np.sqrt(2))*np.array( [ [0,1,0], [1,0,1], [0,1,0] ], dtype='float64') 
sY = (1.0/np.sqrt(2))*np.array( [ [0,-1j,0], [1j,0,-1j], [0,1j,0] ], dtype='complex64') 
sZ = np.array( [ [1,0,0], [0,0,0], [0,0,-1] ], dtype='float64') 


"""
Estructuras de los acoples
""" 
Int1 = np.kron(sZ, np.kron(sZ, sI))  +\
    np.kron(sX, np.kron(sX, sI)) +\
    np.kron(sY, np.kron(sY, sI)) 

Int2 = np.kron(sI, np.kron(sZ, sZ))  +\
    np.kron(sI, np.kron(sX, sX)) +\
    np.kron(sI, np.kron(sY, sY)) 

Int3 =  np.kron(sZ, np.kron(sI, sZ))  +\
    np.kron(sX, np.kron(sI, sX)) +\
    np.kron(sY, np.kron(sI, sY)) 

OZ = np.kron(sZ, np.kron(sI, sI))  +\
    np.kron(sI, np.kron(sZ, sI)) +\
    np.kron(sI, np.kron(sI, sZ)) 

OX = np.kron(sX, np.kron(sI, sI))  +\
    np.kron(sI, np.kron(sX, sI)) +\
    np.kron(sI, np.kron(sI, sX)) 

"""
Exchanges moleculares de los dos tipos de moleculas (3D y 1D)
""" 
J3D = [1.49, 1.49, -0.89]
J1D = [-0.08, -0.08, 0.0]
