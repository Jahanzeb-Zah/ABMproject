#!/usr/bin/python
from mpi4py import MPI
from sklearn.utils import shuffle
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import sys
import os, psutil
T_0 = time.time()
# Initiating MPI Environment
comm = MPI.COMM_WORLD
group = MPI.Group()
# Reading input of number of cores from Input Command
size = comm.Get_size() # new: gives number of ranks/processors to use
rank = comm.Get_rank()
# Population Distribution for Initiation
Sample_20 = [0., 0., 0., 0., 0.]
Sample_20_inf = [1., 0., 0., 0.] 

# Sample to choose from when infecting
Sample_100 = np.zeros(100,dtype = np.float64)
# - Inputting 63% Infection Rate
for a in range(63):
    Sample_100[a] = float(1)
# - Inputting 4% Death Rate
for a in range(0,3):
    Sample_100[a] = float(-999)
# Defining input Data headers
country_id = 'Country ID'
box_id = 'Box ID'
pop_col = 'Population'
density_col = 'Density'
coordinates_col = 'Coordinates'
airport_col = 'Airport'
# Reading Pre-Processed Data
core_data = pd.read_csv('output_5.csv')
# Sorting & Extracting Data
core_data.sort_values(by = [country_id, box_id])
country_data = core_data[country_id].to_numpy(dtype = np.float64)
box_data = core_data[box_id].to_numpy(dtype = np.float64)
pop_data = core_data[pop_col].to_numpy(dtype = np.float64)
density_data = core_data[density_col].to_numpy(dtype = np.float64)
coordinates_data = core_data[coordinates_col].to_numpy()
airport_data = core_data[airport_col].to_numpy(dtype = np.float64)
# Travel Logic initiation
boatSize = 1
boatSize_int = 1
# - Creating Receive buffers in cores for transport
outbound = np.empty(boatSize)
inbound = np.empty(boatSize)
outbound_t = np.empty(boatSize)
inbound_t = np.empty(boatSize)
#Time measurement before allocation of tasks
T_1 = time.time()
# Splitting Population Numbers amongst cores
data = None
if rank == 0:
    data = np.array(pop_data)

# - Creating buffer to receive population data
scatter = np.empty(1)
# - Scattering or Sending Population Data to cores
comm.Scatter(data, scatter, root=0)
# Splitting Country_ID amongst cores
data_c = None
if rank == 0:
    data_c = np.array(country_data)

scatter_c = np.empty(1)
comm.Scatter(data_c, scatter_c, root=0)
# Splitting Box_ID amongst cores
data_b = None
if rank == 0:
    data_b = np.array(box_data)
scatter_b = np.empty(1)
comm.Scatter(data_b, scatter_b, root=0)
# Splitting Density amongst cores
data_d = None
if rank == 0:
    data_d = np.array(density_data)

scatter_d = np.empty(1)
comm.Scatter(data_d, scatter_d, root=0)
# Splitting Airport amongst cores
data_a = None
if rank == 0:
    data_a = np.array(airport_data)

scatter_a = np.empty(1)
comm.Scatter(data_a, scatter_a, root=0)
# Verifying Core Data Distribution
for i in range(size):
    if rank == i:
        core_density = scatter_d

        # Defining & Applying Homogenization Factor
        h_f = 100
        n = int(scatter/h_f)
        a = round(math.sqrt(n))

        while n%a > 0:
        a -= 1
        x_div = a
        y_div = int(n/a)

        # Creating grid of Homogenized Population
        numDataPerRank = n
        epicentre_rank = 532
 # epicentre_rank = 532 @ Huanan Seafood Market

 # Infecting population in Epicentre rank
        if rank == epicentre_rank:
        core_population = np.random.choice(Sample_20_inf,n)
        else:
        core_population = np.zeros(n)
        inf_time = np.zeros(numDataPerRank)

        X_Grid = list(range(0,x_div))
        Y_Grid = list(range(0,y_div))

        # Updating Core Statistics
        inf_count = sum(core_population)
        rate = inf_count/n
        inf_pos = np.where(core_population>=1)

# Initializing Result Variables
send_rate = []
output_arr = np.zeros(1,dtype=np.float64)
column_val = ['Initiation','Country ID','Box ID','Population','Density','Airport']
# Randomization Count Initiation
rand_count = 0
domestic_count = 0
# Defining Time Step Variables
t_step = 100
t_arr = list(range(0,t_step))
inf_out = np.empty(t_step)
imm_out = np.empty(t_step)
vacc_count = 0
# Start of Time Step
for i in range(size):
    t = 0
    for t in range(t_step):

        if rank == i:

            rate = inf_count/len(core_population)
            inf_rate = inf_count/len(core_population)
            send_rate.insert(i,rate)
            inf_pos = np.where(core_population>=1)
            send_rate.insert(i,rate)

            inf_count_1 = inf_count
            inf_rate_1 = inf_rate
            inf_count = sum(1 if x>0 else 0 for x in core_population)
            inf_rate = inf_count/len(core_population)
            imm_count = sum(1 if x<0 and x>-500 else 0 for x in core_population)
            imm_rate = imm_count/len(core_population)
            death_count = sum(1 if x<-500 else 0 for x in core_population)
            death_rate = death_count/len(core_population)
            vacc_rate = vacc_count/len(core_population)

            # Calculating Core Randomization Density Function
            rand_factor = 1/core_density*800

            if t%rand_factor < 1:
                rand_effect = 1
            else:
                rand_effect = 0

            rand_count += rand_effect

            if rand_effect == 1:
                core_population, inf_time = shuffle(core_population,inf_time,random_state = None)

            if t == 0:
                output_arr = np.append(output_arr,[int(scatter_c),rank,int(scatter),float(scatter_d),int(scatter_a)])

            if t == t_step-1:
                column_val.append('Rand Count')
                output_arr = np.append(output_arr,[rand_count])
            inf_out[t]=inf_rate
            imm_out[t]=imm_rate
            output_arr = np.append(output_arr,[t,inf_rate,imm_rate,death_rate,vacc_rate])
            column_val.append('t')
            column_val.append('inf_rate')
            column_val.append('imm_rate')
            column_val.append('death_rate')
            column_val.append('vacc_rate')
            grid_value = 0

        # Start of Core-to-core travel
        if rank == i:
            start = 0
            finish = start+boatSize
            domestic_index = np.where(country_data == scatter_c)
            domestic_list = box_data[domestic_index]
            domestic_indicator = 1

            international_index = np.where(airport_data == 1)
            international_list = box_data[international_index]

            # Travel Restriction Logic
            if t > 3:
                if (inf_count-inf_count_1)>1000 or (inf_rate-inf_rate_1)>=0.05:
                    domestic_indicator = 0
                #print(domestic_count)
            domestic_count += domestic_indicator

            # Core-to-core travel Logic - Domestic

            for a in domestic_list:
                if rank != a:
                    outbound = core_population[start:finish]
                    comm.Send(outbound, dest=int(a))
                    comm.Recv(inbound, source=MPI.ANY_SOURCE)
                    core_population[start:finish] = inbound
                    outbound_t = inf_time[start:finish]
                    comm.Send(outbound_t, dest=int(a))
                    comm.Recv(inbound_t, source=MPI.ANY_SOURCE)
                    inf_time[start:finish] = inbound_t
                    start+=boatSize
                    finish+=boatSize

            # Intra-Core Infection Logic Initiation

            # Vaccination Effect



            newcore_population = core_population.copy()


            if inf_pos[0].size != 0:

                for y in range(y_div):
                    for x in range(x_div):
                        if core_population[grid_value]>0:
                            if inf_time[grid_value] >= 2:
                                inf_time[grid_value] = 0.0
                                newcore_population[grid_value]= -5.0
                            inf_time[grid_value] += 1.0
                        if core_population[grid_value]>=1:
                            if grid_value==0:
                                newcore_population[grid_value+1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value+x_div]+=np.random.choice(Sample_100, 1)
                            elif grid_value==(numDataPerRank-1):
                                newcore_population[grid_value-1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-x_div]+=np.random.choice(Sample_100, 1)
                            elif grid_value == (numDataPerRank-x_div):
                                newcore_population[grid_value+1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-x_div]+=np.random.choice(Sample_100, 1)
                            elif grid_value > (numDataPerRank-x_div):
                                newcore_population[grid_value+1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-x_div]+=np.random.choice(Sample_100, 1)
                            elif grid_value == (x_div-1):
                                newcore_population[grid_value-1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value+x_div]+=np.random.choice(Sample_100, 1)
                            elif grid_value%x_div==0:
                                newcore_population[grid_value+1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value+x_div]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-x_div]+=np.random.choice(Sample_100, 1)
                            elif grid_value < x_div:
                                newcore_population[grid_value+1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value+x_div]+=np.random.choice(Sample_100, 1)
                            else:
                                newcore_population[grid_value+1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-1]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value+x_div]+=np.random.choice(Sample_100, 1)
                                newcore_population[grid_value-x_div]+=np.random.choice(Sample_100, 1)

                        # Vaccination Logic
                        vacc_factor = (100-t)*50
                        if core_population[grid_value]==0 and grid_value%vacc_factor==0:
                            vacc_count+=1
                            newcore_population[grid_value] = -5.0

                        grid_value+=1


            core_population = newcore_population.copy()

            if rank == i and rank in international_list:
                # Core-to-core travel Logic - International
                start = int(len(domestic_list/2))
                finish = start+boatSize
                for b in international_list:
                    if rank != b:
                        outbound = core_population[start:finish]

                        comm.Send(outbound, dest=int(b))
                        comm.Recv(inbound, source=MPI.ANY_SOURCE)
                        core_population[start:finish] = inbound
                        outbound_t = inf_time[start:finish]
                        comm.Send(outbound_t, dest=int(b))
                        comm.Recv(inbound_t, source=MPI.ANY_SOURCE)
                        inf_time[start:finish] = inbound_t
                        start+=boatSize
                        finish+=boatSize
            if t == t_step-1:
                column_val.append('Travel Count')
                output_arr = np.append(output_arr,[domestic_count])

sendbuf = rate
value = np.array(rate,'d')
value_sum = np.array(0.0,'d')
value_max = np.array(0.0,'d')
# perform the reductions:
recvbuf = comm.gather(output_arr, root=0)
comm.Reduce(value, value_sum, op=MPI.SUM, root=0)
comm.Reduce(value, value_max, op=MPI.MAX, root=0)
T_2 = time.time()
T2 = T_2 - T_1
time_value = np.array(T2,'d')
time_value_max = np.array(0.0,'d')
comm.Reduce(time_value, time_value_max, op=MPI.MAX, root=0)
overall_inf_rate = value_sum/size
if rank == 0:
    df = pd.DataFrame(data=recvbuf,columns=column_val)
    df.to_csv('global_normal.csv')
    print("Total Time: ",T_2 - T_0)
    process = psutil.Process(os.getpid())
    print("Memory Used: ",process.memory_info().rss/(1024**2))

print("For Rank: ", rank,", Operation Time: ",T_2 - T_1)



