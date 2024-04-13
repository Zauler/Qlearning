# Scheme of work to find the optimal route using DQN algorithm

"""
Representation of the warehouse 

        *------------------*
        | A   B   C  |  D  
        *----   |    |     |       
          E | F | G     H  |
        |   |   *-----     |
        | I   J    K    L  |
        *--------     -----*
"""

#Import of libraries


import numpy as np




# Parameters and alpha settings for the Q-Learning algorithm
gamma = 0.75
alpha = 0.8 #influences above all the speed of learning 

## DEFINING THE ENVIRONMENT

#Definition of states
location_to_state = {"A":0,
                     "B":1,
                     "C":2,
                     "D":3,
                     "E":4,
                     "F":5,
                     "G":6,
                     "H":7,
                     "I":8,
                     "J":9,
                     "K":10,
                     "L":11}

#Definition of actions
action = [0,1,2,3,4,5,6,7,8,9,10,11]

#Definition of rewards
R = np.array([
    #A,B,C,D,E,F,G,H,I,J,K,L
    [0,1,0,0,0,0,0,0,0,0,0,0],#A
    [1,0,1,0,0,1,0,0,0,0,0,0],#B
    [0,1,0,0,0,0,1,0,0,0,0,0],#C
    [0,0,0,0,0,0,0,1,0,0,0,0],#D
    [0,0,0,0,0,0,0,0,1,0,0,0],#E
    [0,1,0,0,0,0,0,0,0,1,0,0],#F
    [0,0,1,0,0,0,0,1,0,0,0,0],#G 
    [0,0,0,1,0,0,1,0,0,0,0,1],#H
    [0,0,0,0,1,0,0,0,0,1,0,0],#I
    [0,0,0,0,0,1,0,0,1,0,1,0],#J
    [0,0,0,0,0,0,0,0,0,1,0,1],#K
    [0,0,0,0,0,0,0,1,0,0,1,0],#L
    ])

## BUILDING THE SOLUTION WITH Q-LEARNING

#reverse transformation of states to locations
state_to_location = {state : location for location, state in location_to_state.items()}


#A great reward is proposed as far as it is necessary to go.

def route(starting_location, ending_location, priority_list=None):
    R_new = np.copy(R)
    if priority_list:
        r_prior = 500
        priority_states = [location_to_state[i] for i in priority_list ]
        
        for i in priority_states: 
            R_new[:, i] = [r_prior if val > 0 else val for val in R_new[:, i]]
            r_prior -= r_prior / len(priority_list)
        
    ending_state = location_to_state[ending_location]

    R_new[ending_state,ending_state] = 1000
    route = [starting_location]
    
    Q = np.array(np.zeros([12,12]))
    for i in range(0,1000):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state,j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions) #The current state is equal to the future action, because the action is to go to the next state.   
        TD = R_new[current_state, next_state] + gamma*Q[next_state,np.argmax(Q[next_state])]-Q[current_state,next_state]
        Q[current_state,next_state] = Q[current_state,next_state] + alpha*TD
    
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route 

# Print the final route
print("Chosen route: ")

#In this function you can give a start and end location, you can also give the points to pass through sorted by priority, it will return a list of the optimal route. 
print(route("E", "G", priority_list=["K", "C"]))
