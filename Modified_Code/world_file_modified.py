# -*- coding: utf-8 -*-
"""
@author: ABM
"""
# Importing Necessary Libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import geopy.distance
import csv
from pathlib import Path
import os 

# Setting the directory containing all the population data file
cd = Path(__file__).parent
directory = Path(__file__).parent/'files'

files = Path(directory).glob('*')
# Opening output file to write to
output = cd/'output'/'csvOutput.csv'
with open(output,'w',newline='') as f:
    
    # Initializing Global Variables - These will be used to write to country and world ouput files
    country_id = 1
    box_id = 0
    output_arr = []
    
    # Writing Header to Output File
    writer = csv.writer(f)
    header = ['Country ID','Box ID', 'Population', 'Density','Coordinates','Centre Coordinates','Airport','Country Name']
    writer.writerow(header)
   
    # Defining Time to track Computation Time
    T_0 = time.time()
    
    # Start of loop to perform operation on all files in directory
    for file in files:
    # Time measurement before allocation of tasks
        T_1 = time.time()
        country_box_id = 0
        
        # Defining Column headers in data
        lat_col = 'latitude'
        lon_col = 'longitude'
        pop_col = 'population'

        # Reading file at path - doesn't open file as csv
        population = pd.read_csv(file)

        # Sorting File based on Latitude & Longitude into ascending order
        population.sort_values(by = [lat_col, lon_col])


        # Extracting the data into arrays to perform operations
        lat = population[lat_col].values
        lon = population[lon_col].values
        pop_count = population[pop_col].values

        # Calculation & Verifying Country Population
        pop_total = sum(pop_count)
        print(file,"Total Population: ",pop_total)

        # Logic to assign only one core to Countries with a population less than 1 M
        if pop_total < 1000000:
            core_count = pop_total
        else:
            core_count = 1000000

        # Calculating Variables used in Core Distribution
        n = int(pop_total/core_count)

        # if country_id in (33,69):
        #     n = int(n/4)

        # if country_id in (70,104):
        #     n = int(n/2)
       
        # Output Number of Cores - Validation Check

        # Logic to determine closest multiples of number of cores
        a = round(math.sqrt(n))
        while n%a > 0:
            a -= 1

        # Prime number mitigation
        if a == 1 and n>1:
            n -= 1

        a = round(math.sqrt(n))

        while n%a > 0:
            a -= 1

        core_number = n

        print("Total Number of Cores: ", core_number)
        
        # Defining limits of map (in coordinates)
        lat_min = min(lat)
        lat_max = max(lat)

        lon_min = min(lon)
        lon_max = max(lon)

        # Calculating Core distribution parameters
        lat_diff = lat_max - lat_min
        lon_diff = lon_max - lon_min
        
        if lat_diff > lon_diff:
            # n/a is larger than a
            lat_n = int(n/a)
            lon_n = a
        else:
            lon_n = int(n/a)
            lat_n = a

        x_lin_width = lon_diff/lon_n
        y_lin_width = lat_diff/lat_n

        alpha_lat = 1
        
        # Upper limit of y range
        delta_lat = alpha_lat*y_lin_width
        
        # Lower limit of y range
        delta_lat_1 = (alpha_lat-1)*y_lin_width

        alpha_lon = 1
        # Upper limit of x range
        delta_lon = alpha_lon*x_lin_width
        # Lower limit of x range
        delta_lon_1 = (alpha_lon-1)*x_lin_width

        # Initializing Variables for core-distributed population values
        # Each element is the population of that box in latitude order
        box_popu_list = []
        total_box_popu = 0
        lat_list = []
        lon_list = []
        density = []
        box_height = 0
        box_width = 0
        box_area = []
        coordinates = []
        centre_coordinates = []
        box_density = 0
        index_val = np.where(lat<=lat_max)
        non_empty_core = 0

        # Starting loop for each core (a), and hence distributing population
        for a in range(1,core_number+1):
            
            index_val_x = np.where(np.logical_and(lon>=(lon_min+delta_lon_1),lon<=(lon_min+delta_lon)))
            index_val_y = np.where(np.logical_and(lat>=(lat_min+delta_lat_1),lat<=(lat_min+delta_lat)))
            
            #All the indices which are for the current core we are looping over
            core_index = np.intersect1d(index_val_x,index_val_y)
           
            #pop_count[core_index]

            for b in core_index:
                total_box_popu+=pop_count[b]

            # Coordinates of Core
            point_a = (lat_min+delta_lat_1, lon_min+delta_lon_1)
            point_b = (lat_min+delta_lat_1, lon_min+delta_lon)
            point_c = (lat_min+delta_lat,lon_min+delta_lon_1)
            point_d = (lat_min+delta_lat, lon_min+delta_lon)
            box_points = [point_a,point_b,point_c,point_d]
            point_centre = (((lat_min+delta_lat_1+lat_min+delta_lat)/2),((lon_min+delta_lon_1+lon_min+delta_lon)/2))

            # Calculating Core Area & Density
            box_height = geopy.distance.distance(point_a, point_c).km
            box_width = geopy.distance.distance(point_a,point_b).km
            box_density = total_box_popu/(box_height*box_width)

            # Appending Data to Arrays
            box_area.append(box_height*box_width)
            density.append(box_density)
            box_popu_list.append(int(total_box_popu))
            coordinates.append(box_points)
            centre_coordinates.append(point_centre)

            # Calculating Non-Empty Cores
            if total_box_popu>=100:
                non_empty_core += 1

            # Re-initialazing & Updating Variables for Next Run
            total_box_popu = 0

            alpha_lon += 1
            # Upper x limit
            delta_lon = alpha_lon*x_lin_width
            # Lower x limit
            delta_lon_1 = (alpha_lon-1)*x_lin_width

            # If traversed multiple of longitude n into core number increment latitude and reset longitude 
            if a%lon_n == 0:
                alpha_lat += 1
                delta_lat = alpha_lat*y_lin_width
                delta_lat_1 = (alpha_lat-1)*y_lin_width
                alpha_lon = 1
                delta_lon = alpha_lon*x_lin_width
                delta_lon_1 = (alpha_lon-1)*x_lin_width
       
        # Calculating Statistics
        avg_area = sum(box_area)/len(box_area)
        avg_density = core_count/avg_area
        density = np.array(density)
        box_heavy = np.where(density > avg_density)
        box_light = np.where(density < avg_density)

        # Output Country Stats - Validation
        if non_empty_core > 10:
            print("Number of non-empty Cores: ",non_empty_core)
            print("Number of Empty Cores: ",len(box_popu_list)-non_empty_core)
            print("Population Verification: ",sum(box_popu_list))
            print("Minimum Core Count: ", min(box_popu_list))

        print("Maximum Core Count: ",max(box_popu_list))
        print("Maximum Density (per kmsq): ", np.max(density))

        # Defining Airport Cores
        airport_list = np.zeros(core_number)
        airport_index = np.argmax(box_popu_list)
        airport_list[airport_index] = 1

        # Find country name
        fileName = os.path.basename(file)
        partialCountryName = fileName.split('.')[0]
        countryName = partialCountryName.split('_')[0]

        # Write to output file 
        for a in range(core_number):
            if box_popu_list[a] > non_empty_core*100*2:
                output_row =[float(country_id),float(box_id),float(box_popu_list[a]),density[a],coordinates[a],centre_coordinates[a],float(airport_list[a]),countryName]
                writer.writerow(output_row)
                box_id += 1
                

        # Printing Computation Time for country - Validation
        T_5 = time.time()
        T_delta = T_5-T_1
        print("__________________________________________________________")
        print("Total Time: ",T_delta)
        print("__________________________________________________________")
    
    # Printing Global Computation Time - Validation
    T_delta = T_5-T_0
    print("Total Processing Time: ",T_delta)
   
    # Closing File
    # Ploting Country Core Distribution & Saving Images to Folder
    #plot_arr = np.flipud(np.reshape(box_list, (lat_n,lon_n)))
    #plt.matshow(plot_arr,cmap=plt.get_cmap('coolwarm'))
    
    #For current country file strip name 
    # fname = str(file).rstrip(".csv")
    # fname = fname.lstrip("files")
    # country_name = fname.lstrip("\_0123456789")
    # country_name = country_name.rstrip("\_0123456789")
    #print(country_name)


    # fname = "data"+fname
    #plt.savefig(fname)
    #plt.close()

    # fname = fname+".csv"
    # df = pd.DataFrame(header)
    # df.to_csv(fname)

    # Writing Country Data to Global Output File
    # country_f = open(fname,'w',newline='')
    # mywriter = csv.writer(country_f)
    # mywriter.writerow(header)
    

    # country_id += 1
    # country_f.close()
