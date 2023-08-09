# ABMproject
This repo is a home for an Agent Based Model (ABM) to forecast the spread of Covid-19.  
The model is parallelised to run on an arbitrary number of cores, using the Message Passing Interface (MPI).  
The input to the model is created using the World_Discretisation.py script. This script aggregates the population of a given geographical area.  
The Parallel_Model.py script runs the ABM on the file created by the population discretisation script.  
