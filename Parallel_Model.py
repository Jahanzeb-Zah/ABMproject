# Import libraries 
from mpi4py import MPI
import numpy as np
import pandas as pd
from pathlib import Path
import synthpops as sp
import sciris as sc
import sys 
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define global functions
def myFun (*args):
    '''
    Convenience function that prints arguments. First argument must be rank.
    '''
    args = list(args)
    _, *objs =  args
    print(f'rank is {args[0]}', *objs)

def getContacts (agent):
    '''
    Returns value of contacts key --> nested dict
    '''
    return networkedPop[int(agent)]['contacts']

def infect (agentContacts:list,t:int,contactLayer:str)-> None:
    '''
    Agents passed in are contacts of infectious agent. Runs transmission model on contacted agents. 
    '''
    for agent in agentContacts:
        if agentState[agent] == 0:
            # Calculate transmission prob for contacted agent using age and contact setting
            age = agentAge[agent]   
            ind = np.digitize(age,ageBins,right=False)-1
            agentrSurcept = rSurcept[ind]
            prob = clTransProb[contactLayer]*agentrSurcept
            # If tranmission event occurs, change state and recovery time
            if random.uniform(0,1) <= prob:
                agentState[agent] = 1
                recoveryLength = int(np.random.lognormal(2.2,0.188,1))
                infTime[agent] = t + recoveryLength
                transEvn[clTransEnvIndex[contactLayer]] += 1
              
    return None

def recover (agent,t):
    '''
    Agent passed is recoved after infection. Recovered agent is set as immune or dead. 
    '''
    age = agentAge[agent]   
    ind = np.digitize(age,ageBins,right=False)-1               
    # Set state as dead using probability from age bin
    if random.uniform(0,1) <= pDeath[ind]:
        agentState[agent] = 2
    # If death event didn't happen, set state as immune
    else:
        agentState[agent] = 3
        infTime[agent] = t + int(np.random.normal(3*30,10,1))
    return None

def createLocationFile (distArrayName, countryName) :
    '''
    Create a new location file in the synthpops data dir using usa values.
    Only the population age distribution list is edited. 
    '''
    input_location_filepath = "usa.json"
    output_location_filepath = Path(sp.default_datadir_path())/(countryName +'.json')
    # Loads the default location object from usa file path 
    location_data = sp.load_location_from_filepath(input_location_filepath)
    location_data.population_age_distributions[2].distribution = distArrayName
    sp.save_location_to_filepath(location_data, str(output_location_filepath.resolve()))
    return None 


def sendAgents(coreToReceive):
    '''
    Runs travel model on given receiving core
    '''
    agentsToSend = np.random.choice(np.flatnonzero(agentState!=2),
                                    agentSendCount,replace=False)
    outboundState = agentState[agentsToSend]
    outboundTime = infTime[agentsToSend]
    outboundAge = agentAge[agentsToSend]
    outbound = np.vstack((outboundState,outboundTime,outboundAge))
    myFun(rank,'time step',t, 'message sending',coreToReceive)
    comm.Send(outbound, dest=int(coreToReceive))
    myFun(rank,'time step',t, 'message sent',coreToReceive)
    # We know the number of agents we'll receive (agentSendCount) but not positions of 
    # those agents in sender array
    comm.Recv(inbound, source=MPI.ANY_SOURCE)
    inboundState, inboundT, inboundAge = inbound[0,:], inbound[1,:], inbound[2,:]
    infTime[agentsToSend] = inboundT
    agentState[agentsToSend] = inboundState
    agentAge[agentsToSend] = inboundAge
    agentAgeBin[agentsToSend]= np.digitize(inboundAge,ageBins,right=False)
    myFun(rank,'time step',t, 'message received')

    return None

def binStateCount(ageBin:int, state:int) ->int:
    '''
    Number of agents with the given agenBin and state. Used in allBinState func. 
    '''
    binAgentStates = agentState[agentAgeBin==ageBin]
    agentStateCount = np.count_nonzero(binAgentStates==state)
    if np.count_nonzero(agentAgeBin== ageBin) > 0:
        agentStateProp = agentStateCount/np.count_nonzero(agentAgeBin== ageBin)
    else:
        agentStateProp = 0
    return agentStateProp

def allBinsSate()-> list:
    '''
    Returns list of count of agents for each state ordered by bin
    '''
    states =[]
    for bin in range(1,len(ageBins)+1):
        for state in np.array([0,1,2,3]):
            states.append(binStateCount(bin,state))
    return states

def stateByBinHeaders():
    '''
    Returns list of colmnVal headers for each agent state ordered by agebin
    '''
    states = ['susceptProp','infProp','deathProp','immProp',]
    headers = []
    for bin in range(1,len(ageBins)+1):
        for state in states:
            headers.append(state+'_Bin_'+str(bin))
    return headers

boxesPerCore = 1
hf = 100
# Population age distribution arrays used to create Location files
distribution_niger = [[0, 4, 4379012],
 [5, 9, 4379012],
 [10, 14, 2950563],
 [15, 19, 2950563],
 [20, 24, 1858679],
 [25, 29, 1858679],
 [30, 34, 1129510],
 [35, 39, 1129510],
 [40, 44, 764348],
 [45, 49, 764348],
 [50, 54, 523424],
 [55, 59, 523424],
 [60, 64, 322375],
 [65, 69, 322375],
 [70, 74, 147980],
 [75, 79, 147980],
 [80, 84, 6858],
 [85, 89, 6858],
 [90, 94, 6858],
 [95, 100, 6858]]
distribution_japan  = [[0,4,5089984],
 [5, 9, 5089984],
 [10, 14, 5633586],
 [15, 19, 5633586],
 [20, 24,  6073662],
 [25, 29,  6073662],
 [30, 34, 7227708],
 [35, 39, 7227708],
 [40, 44, 9236539],
 [45, 49, 9236539],
 [50, 54, 8270758],
 [55, 59, 8270758],
 [60, 64, 7937618],
 [65, 69, 7937618],
 [70, 74, 8092688],
 [75, 79, 8092688],
 [80, 84, 5675688],
 [85, 89, 1891896],
 [90, 94, 1891896],
 [95, 100, 1891896]
]


# read and extract from input file
if rank == 0:
    cwd = Path.cwd()    
    worldOutput = cwd/"Input"/"worldOutputData8EqualPopu.csv"
    resultsDir = cwd/"Testing"    
    core_data = pd.read_csv(worldOutput)
    country_id = 'Country ID'
    box_id = 'Box ID'
    pop_col = 'Population'
    countryNameCol = 'Country Name'
    density_col = 'Density'
    airportCol  = 'Airport'
    demographicCol = 'Demographic Bin'
    countryName = list(core_data[countryNameCol])
    box_data = core_data[box_id].to_numpy(dtype='i')
    core_data[country_id] = core_data[country_id]
    country_data = core_data[country_id].to_numpy(dtype = 'i')
    # Apply homogenisation factor 
    core_data[pop_col] = core_data[pop_col]/hf
    pop_data = (core_data[pop_col].to_numpy(dtype = 'i'))
    #pop_data = pop_data/100
    density_data = core_data[density_col].to_numpy(dtype='f')
    airport_data = core_data[airportCol].to_numpy(dtype = 'i')
    age_dist_bin_data = core_data[demographicCol].to_numpy(dtype = 'i')
    columnVal = ['Initiation','Country ID','Box ID','Population','Density','Airport', 'Demographic Bin']
    finalStates = ['Transmissions']
    # Create Location files for demographics
    createLocationFile(distribution_japan,'japan')
    createLocationFile(distribution_niger,'niger')



else:
    pop_data = None
    country_data = np.empty(size,dtype = 'i' )
    airport_data = np.empty(size,dtype = 'i' )
    density_data = np.empty(size,dtype='f')
    age_dist_bin_data = np.empty(size,dtype = 'i' )

population = np.empty(boxesPerCore,dtype = 'i' )

comm.Scatter(pop_data, population, root=0)

comm.Bcast(country_data, root=0)
comm.Bcast(density_data,root=0)
comm.Bcast(airport_data,root=0)
comm.Bcast(age_dist_bin_data,root=0)

'''
Create a Networked popu on each core in condensed format
'''
locationFileMapping = ['niger','usa','japan']
contactMatrixMappings = ['Niger', 'United States of America','Japan']

# Parameters of networked popu
pars = sc.objdict(
    # Access the integer inside of population array
    n                               = population[0],
    rand_seed                       = 0,
    country_location                = locationFileMapping[age_dist_bin_data[rank]],
    sheet_name                      = contactMatrixMappings[age_dist_bin_data[rank]],
    household_method                = 'infer ages',
    smooth_ages                     = 1,
    with_school_types               = 0,
    use_default                     = True
)

# Create networked popu
networkedPop = sp.Pop(**pars).to_dict()

# convert each agents workplace contacts to integers 
for key,value in networkedPop.items():
    networkedPop[key]['contacts']['W']= list(map(int,networkedPop[key]['contacts']['W']))

# convert agent key to integer
networkedPop = {int(key):value for key, value in networkedPop.items()}

'''
Disease Parameters
'''
# Initialise agent state and infection time arrays 
initialiseInfProb = np.array([1,0,0,0],dtype=np.int64)
probabilityInitialise = np.array([0],dtype=np.int64)

pDeath = np.array([0.00002,0.00002,0.0001,0.00032,0.00098,
                   0.00265,0.00766,0.02439,0.08292,0.16190])*2
rSurcept = np.array([0.34,0.67,1,1,1,1,1,1.24,1.47,1.47])

# infection probability in each contact layer
comInf = 0.005
workInf,schInf,housInf = 2*comInf,2*comInf,10*comInf
beta = (comInf+2*workInf+2*schInf+10*housInf)/15
clTransProb = {'H':housInf,'S':schInf,'W':workInf,'C':comInf}

# Immunity and Infectious Period distribution parameters
# Mu and sigma of lognormal dist
mLn = 9
sLn = np.sqrt(2.9)
# Mu and sigma of underlying normal dist
mu = np.log(mLn)-(sLn**2)/2
sd = np.sqrt(np.log(1+(sLn/mLn)**2))

# Initialise agent state and infection time arrays for all cores 
epicentreRank = 0
# infTime is the time step at which the immunity/infectivity period ends
infTime = np.random.choice(probabilityInitialise,population[0])

# Create array of agents age
agentAge = np.empty(len(networkedPop),dtype=np.int64)
for i in range(len(networkedPop)):
    agentAge[i] = networkedPop[i]['age']

# Age bin each agent falls into
ageBins = np.arange(0,100,10)
agentAgeBin = np.digitize(agentAge,ageBins,right=False)

# initialise infectious agents in epicentre rank and their recovery time
if rank == epicentreRank:
    agentState = np.random.choice(initialiseInfProb,population[0])
    for agent in np.flatnonzero(agentState==1):
        recoveryLength = int(np.random.lognormal(2,sd,1))
        infTime[agent] = recoveryLength
else:
    agentState = np.random.choice(probabilityInitialise,population[0])


'''
Travel Initialisation Variables
'''

# Find domestic cores of this rank
coreCountryID = country_data[rank]
domesticCores = np.flatnonzero(country_data==coreCountryID)
# International cores indices
airportCores = np.flatnonzero(airport_data==1)
# Number of agents to send per step
agentSendCount = 1*10
# Buffer array for agent travel
inbound = np.empty((3,agentSendCount), dtype =np.int64)


# 100 time steps
tsteps = 104
time = np.arange(tsteps)

'''
Contact Layer Initialisation Variables
'''
coreDensity = density_data[rank]
commDensityMax = np.max(density_data)
commDensityMin = np.min(density_data)
communityGroupSize = 5

# Maximum proportion of core population to include in community contact layer
maxP = 0.25

corePopu = np.arange(0,population[0])
normalisedDensity = (coreDensity-commDensityMin)/(commDensityMax-commDensityMin)
# Count of agents in community layer. Varies between cores
cLayerAgentsCount = round(normalisedDensity*population[0]*maxP)
# Count of agents in remainder group
r =  cLayerAgentsCount%communityGroupSize
# Quotient to find number of full community groups 
communityGroupCount = cLayerAgentsCount//communityGroupSize


# Output array
output = np.empty(1)
output = np.append(output, [country_data[rank],rank,population[0]*hf,coreDensity,airport_data[rank],age_dist_bin_data[rank] ])

# Recording transmission events
clTransEnvIndex = {'H':0,'S':1,'W':2,'C':3}
transEvn = np.array([0,0,0,0])

# TODO: Remove 
myFun(rank, 'DemographicBin', age_dist_bin_data[rank],'BinPopu',agentAgeBin )
'''
Main Model Loop
'''
for t in time:

    '''
    Generate Community Contacts
    '''
    # Agents in community layer
    clAgents = np.random.choice(corePopu,cLayerAgentsCount,replace = False)
    # If a remainder group exists
    if r !=0:
        # List of agents in remainder group
        rGroup = list(clAgents[0:r])
        for i,a in enumerate(rGroup):
            # Makes new copy of remainder group
            rGroupCopy = rGroup.copy()
            # Remove current agent from copied group
            rGroupCopy.pop(i)
            # Add copied group with deleted agent to current agent's contacts
            networkedPop[a]['contacts']['C'] = rGroupCopy

    for commuGroup in range(0,communityGroupCount):
    # Create slice of agents with size equal to community group
        currentGroup = list(clAgents[commuGroup*communityGroupSize+r:communityGroupSize*(commuGroup+1)+r])
        for i,a in enumerate(currentGroup):
            currentGroupCopy = currentGroup.copy()
            currentGroupCopy.pop(i)
            # Add community group to current agent's contacts
            networkedPop[a]['contacts']['C']= currentGroupCopy

    '''
    Record Simulation State in Output Arrays
    '''
    susceptProp = np.count_nonzero(agentState==0)/population[0]
    infProp = np.count_nonzero(agentState==1)/population[0]
    deathProp = np.count_nonzero(agentState==2)/population[0]
    immProp = np.count_nonzero(agentState==3)/population[0]
    output = np.append(output, [t,susceptProp, infProp, deathProp, immProp,]+[transEvn[clTransEnvIndex['H']]*hf,transEvn[clTransEnvIndex['S']]*hf,
                                transEvn[clTransEnvIndex['W']]*hf,transEvn[clTransEnvIndex['C']]*hf,]+allBinsSate())

    if rank == 0:
        headers = ['time','susceptProp','infProp','deathProp','immProp']+['tranEvnH','tranEvnS','tranEvnW','tranEvnC']+stateByBinHeaders()
        columnVal.extend(headers)
    
    
    '''
    International travel model
    '''
    if rank in airportCores:
        for a in airportCores:
            if rank == a:
                pass
            else:
                myFun(rank,'time step',t, 'about to send int')
                sendAgents(a)
                myFun(rank,'time step',t, 'finished send/receive')
    
    
    '''
    Domestic Travel Model
    '''
    for a in domesticCores:
        if rank == a:
            pass
        else:
            myFun(rank,'time step',t, 'about to send dom')
            sendAgents(a)
            myFun(rank,'time step',t, 'finished send/receive')
    
    '''
    Infection Model
    '''
    # returns index of true elements. returned object is an array not a tuple.
    currentInfGen = np.flatnonzero(agentState==1)
    # Check if there are any currently infected agents
    if np.size(currentInfGen) >0:
        # Iterate over every infected agent
        for agent in currentInfGen:
            # If agent no longer infectious, attribute as dead or immune 
            if infTime[agent] == t:
                recover(agent,t)
            else:
                # Find contacts of current infectious agent
                agentAllContacts = getContacts(agent)
                agentHContacts = agentAllContacts['H']
                agentSContacts = agentAllContacts['S']
                agentWContacts = agentAllContacts['W']
                agentCContacts = agentAllContacts['C']
                
                # Infect each household contacted agent
                if bool(agentHContacts) == True:
                     infect(agentHContacts,t,'H')
                
                # Infect each school contacted agent
                if bool(agentSContacts) == True:
                   infect(agentSContacts,t,'S')
                   
                # Infect each workplace contacted agent
                if bool(agentWContacts) == True:
                    infect(agentWContacts,t,'W')
                
                # Infect each community contacted agent
                if bool(agentCContacts) == True:
                    infect(agentCContacts,t,'C')
    
    # Check for end of immunity period 
    currentT = np.flatnonzero(infTime==t)
    for agent in currentT:
        if agentState[agent] == 3:
            # If time array equals current time and state is immune, set as susceptible
            agentState[agent] == 0

recvbuf = comm.gather(output, root=0)        

if rank == 0:
    simulationOutput = resultsDir/'parallelOverallModel.csv'
    with open(simulationOutput,'w', newline = '') as s:
        df = pd.DataFrame(data=recvbuf,columns=columnVal)
        df.to_csv(path_or_buf= s, index = False)

 
                                                      