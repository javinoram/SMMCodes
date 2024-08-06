# Codes

Here we have two diferents folder, both are magnetocaloric codes but differ in the number of molecules.

In each one, we allow two types of values from the intramolecular exchange values (frustated values and linear values).

The coments in the code are in spanish, sorry for that.

## One molecule
To execute the code you can use *python3 file.py struct hx*.

For *struct* you can choose between *3D* and *1D*. And, *hx* es the value of the X magnetic field, is a float value.

The *hx* parameter is not needed in the *Iso.py*, *Mag.py* and *SHeat.py* files, the hx parameter is defined inside. This 
parameter is needed for the *Atc.py* file.

## Two molecules
To execute the code you can use *python3 file.py struct conf J hx*.

For *struct* you can choose between *3D* and *1D*, for *conf* you can choose between 1,2 and 3, for *J* is the 
intermolecular exchange, is a float value. And, *hx* es the value of the X magnetic field, is a float value.


