#  AI-agent who can learn Pacman through feature based Q-learning and reinforcement learning

# FEATURES :
#------------------------------------
# Distance to closest normal ghost: x tiles away
# Distance to closest vulnerable ghost: x tiles away
# Distance to closest pellets: x tiles away
# Distance to closest food pellets: x tiles away

# ACTIONS : # RIGHT # LEFT # DOWN # UP

############    Q-TABLE SET-UP  #################
#                           R   L   D   U
#   normal ghost            x   x   x   x
#   vulnerable ghost        x   x   x   x
#   pellet                  x   x   x   x
#   power pellet            x   x   x   x
#
#################################################

# REWARDS :
#----------------------
# Eating a pellet:
# Killing a ghost:
# Losing life/Dying:
# Revisit previous/empty positions:

from pacman import *
import random
import numpy as np 
import math
import copy

# global constants
NUM_FEATURES = 16    
NUM_STATES = 625        # 2**NUM_FEATURES 
NUM_ACTIONS = 4         # RIGHT, LEFT, DOWN or UP 
ACTIONS = ['RIGHT', 'LEFT', 'DOWN', 'UP'] 
LEARNING_RATE = 0.5     # alpha [0,1]
EXPLORATION_RATE = 0.0  # [0,1]
DISCOUNT = 0.8          # gamma [0,1]

Q_table = np.zeros([NUM_STATES, NUM_ACTIONS])
features = np.zeros([NUM_FEATURES]).astype(int) # binary

# global variables
previousState = -1
previousAction = 'LEFT'
prepreAction = 'DOWN'
currentPelletCount = 140
hitByGhost = 0
ateGhost = 0
towardsPellet = 0
returnCounter = 0
oldFeatures = np.zeros([NUM_FEATURES]).astype(int)

aiTraining = 1
deaths = 0

#####################   FEATURE DEFINITIONS    ######################
#           COMPUTE GHOST/PELLET/ENERGIZER DISTANCE                 #
#####################################################################

# Compute distance to the closest ghost in given direction
#   1 = normal
#   2 = vulnerable
#   (3 = spectacles)
def closest_ghost_dir(state = 1):

    posX = (player.x) / TILE_HEIGHT
    posY = (player.y) / TILE_WIDTH
    minDist = np.inf
    closestGhostPath = ""
    idx = 0
    
    for i in range(0, 4, 1):
        if ghosts[i].state == state:

            row = ghosts[i].nearestRow 
            col = ghosts[i].nearestCol 
            
            # only get ghost path when ghost is close enough
            manhattanDist = abs(player.nearestRow - row) + abs(player.nearestCol - col)
            # print('manhattan: ', manhattanDist)
            if (manhattanDist) < 8:
                distance = path.FindPath((player.nearestRow, player.nearestCol), (row,col)) 
                if distance != False and len(distance) > 0 and len(distance) < minDist:
                    minDist = len(distance)
                    closestGhostPath = distance[0]
                    #print('shortest path to Ghost of state ' , state ,': ' , closestGhostPath)
                    idx = translateChar(closestGhostPath)
            else: 
                continue
    
    # no close ghost found
    if closestGhostPath == "":
        return '', minDist
    return ACTIONS[idx], minDist


#														x	x	x	x	x	x	x
#							x	x	x	x	x			x	-	-	-	-	-	x		
#		x	x	x			x	-	-	-	x			x	-	-	-	-	-	x
#		x	o	x			x	-	o	-	x			x	-	-	o	-	-	x
#		x	x	x			x	-	-	-	x			x	-	-	-	-	-	x
#							x	x	x	x	x			x	-	-	-	-	-	x
#														x	x	x	x	x	x	x
#
# Loop beyond 3x3 grid around pacman to find closest remaining pellets
# looking 1 row and col further at each iteration
def find_far_off_pellet(playerRow,playerCol):
    paths = {}
    radius = 1
    offset = 1
    idx = 5

    while paths == {}:
       
        # top and bottom row
        for tileRow in [playerRow -(radius + offset), (playerRow + radius + offset)]:  # [-2,2]
            for tileCol in range(playerCol - (radius + offset), (playerCol + radius + offset + 1)):  # [-2,-1,0,1,2]
                # find paths
                
                if tileRow > 0 and tileRow < 21 and tileCol > 0 and tileCol < 18: 
                    #print('top/bottom: ', tileRow, tileCol)
                    paths.update( find_path((tileRow, tileCol),(playerRow, playerCol)) )
                
        # left and right columns
        for tileRow in range(playerRow - (radius + offset), (playerRow + radius + offset + 1)):  # [-2,-1,0,1,2]
            for tileCol in [playerCol -(radius + offset), (playerCol +radius + offset)]:  # [-2,2]
                # find paths 
                
                if tileRow > 0 and tileRow < 21 and tileCol > 0 and tileCol < 18: 
                    #print('right/left: ', tileRow, tileCol)
                    paths.update( find_path((tileRow, tileCol), (playerRow, playerCol)) )
                
        offset += 1
          
    # get first char of paths with minimum distance to pellet
    minval = min(paths.values())
    bestActions = [k[0] for k, v in paths.items() if v==minval]

    # one shortest path
    if len(bestActions) == 1: 
        idx = translateChar(bestActions[0])         

    # several paths with same distance - pick random
    else:
        idx = translateChar(random.choice(bestActions))

    return idx

# helper function to find_far_off_pellet    
def find_path(tile, pacman):
    paths = {}
    tileRow, tileCol = tile
    playerRow, playerCol = pacman
    posY = (player.y) / TILE_WIDTH  # Pacman absolute position
    posX = (player.x) / TILE_HEIGHT

    # if tile contains pellet, get path and compute distance 
    if (thisLevel.GetMapTile(tile) == tileID['pellet']):
        pelletPath = path.FindPath(pacman, tile) 
                                      
        # store all possible paths with their distance to Pacman
        if pelletPath != False and len(pelletPath) > 0: #and translateChar(pelletPath[0]) in possibleActions:
                            
            # compute actual distance to pellet & store path in dictionary
            distance = abs(posY - tileRow) + abs(posX - tileCol)                            
   
            # Check if distance is already in the dictionary and if it exists check which distance is lower
            if pelletPath not in paths:
                paths.update( {pelletPath: distance} )
            else:
                if paths[pelletPath] > distance:
                    paths.update({pelletPath: distance})
        
    return paths
    
# Loop over Pacmans closest 3x3 grid to find shortest path to pellets
def find_pellet_paths(playerRow, playerCol, radius, pelletType):
    possibleActions = get_possible_actions()    # 0, 1, 2, 3    R L D U
    paths = {}

    posY = (player.y) / TILE_WIDTH  # Pacman absolute position
    posX = (player.x) / TILE_HEIGHT

    # loop over grid around Pacman
    for tileRow in range(playerRow - radius, playerRow + radius + 1):           # y 
        for tileCol in range(playerCol - radius, playerCol + radius + 1):       # x
                
            # if tile contains pellet, get path and compute distance 
            if (thisLevel.GetMapTile((tileRow, tileCol)) == tileID[pelletType]):

                pelletPath = path.FindPath((playerRow, playerCol), (tileRow,tileCol)) 
                              
                # store all possible paths with their distance to Pacman
                if pelletPath != False and len(pelletPath) > 0 and translateChar(pelletPath[0]) in possibleActions:
                            
                    # compute actual distance to pellet & store path in dictionary
                    distance = abs(posY - tileRow) + abs(posX - tileCol)                            

                    # Check if distance is already in the dictionary and if it exists check which distance is lower
                    if pelletPath not in paths:
                        paths.update( {pelletPath: distance} )
                    else:
                        if paths[pelletPath] > distance:
                            paths.update( {pelletPath: distance} )
    # print(paths)
    return paths

# compute closest distance between pellet and pacmam
def closest_pellet_dir(radius = 1, pelletType = 'pellet'):

    posY = (player.y) / TILE_WIDTH  # Pacman absolute tile position
    posX = (player.x) / TILE_HEIGHT

    playerRow = int(round(posY)) # Pacman rounded tile - dummys for start
    playerCol = int(round(posX))

    diffX = posX - playerCol   # +: RIGHT, -: LEFT  
    diffY = posY - playerRow   # +: DOWN, -: UP 

    paths = {}
    idx = 5
           
    # Pacman is in centre of a tile, find possible paths to pellet from tile
    if diffY == 0 and diffX == 0:
        paths = find_pellet_paths(playerRow, playerCol, radius, pelletType)
 
    # Pacman is in between 2 tiles, check for possible paths for both the tiles
    else:
        #Optimize?
        for i in range(0,2):                      
            if(diffY != 0 and i != 1):
                playerRow = math.ceil(posY)         
            elif(diffY != 0):
                playerRow = math.floor(posY)        

            if(diffX != 0 and i != 1):              
                playerCol = math.ceil(posX)
            elif(diffX != 0):
                playerCol = math.floor(posX)

            paths.update( find_pellet_paths(playerRow, playerCol, radius, pelletType) )
        
    # Pellet paths found
    if len(paths) > 0:
  
        # get first char of paths with minimum distance to pellet
        minval = min(paths.values())
        bestActions = [k[0] for k, v in paths.items() if v==minval]
        
        # one shortest path
        if len(bestActions) == 1: 
            idx = translateChar(bestActions[0])
                
        # several paths with same distance -- pick random
        else:
            # print('Picked random move')
            idx = translateChar( random.choice(bestActions) )
    
    # no close pellet found
    else:
        if pelletType == 'pellet': 
            idx = find_far_off_pellet(playerRow,playerCol)
            #print('closest faroff pellet: ', ACTIONS[idx])
        
        # don't look further for power pellets
        else:
            return ''
  
    # print('Closest ', pelletType, ' dir: ', ACTIONS[idx])
    return ACTIONS[idx]



#####################   FEATURE & REWARD COMPUTATIONS    ####################
#       BINARY RESULTS FOR EACH FEATURE & POS/NEG REWARDS FOR ACTIONS       #
#############################################################################
#
# features[0-3]: will be 1 if a normal ghost is in direction RLDU
# features[4-7]: will be 1 if vulnerable ghost is close
# features[8-11]: will be 1 if food pellet is close
# features[12-15]: will be 1 if power pellet is close

# features[?]: will be 1 if time of closest vulnerable ghost is short
def calculate_features(pos = []):
    idx = 0   
    GHOSTCLOSE = 120 # ~ 5 tiles away    

    # normal ghost distance for every action
    ghostDirection, ghostDist = closest_ghost_dir()
    for action in ACTIONS:
        if ghostDist < GHOSTCLOSE and action == ghostDirection:
            features[idx] = 1 
        else:
            features[idx] = 0
        idx += 1

    # vulnerable ghost distance for every action
    vghostDirection, vghostDist = closest_ghost_dir(2)  # state = 2
    for action in ACTIONS:
        if vghostDist < GHOSTCLOSE and action == vghostDirection:
            features[idx] = 1 
        else:
            features[idx] = 0
        idx += 1

    # food pellet distance for every action
    # print('   call closest_pellet_dir')
    pelletDirection = closest_pellet_dir()
    for action in ACTIONS:
        if pelletDirection != '' and action == pelletDirection:
            features[idx] = 1
        else:
            features[idx] = 0
        idx += 1


    # power pellet distance for every action
    ppelletDirection = closest_pellet_dir(1, 'pellet-power')
    for action in ACTIONS:
        if ppelletDirection != '' and action == ppelletDirection:
            features[idx] = 1
        else:
            features[idx] = 0
        idx += 1

# make unique states for each possible feature configuration
def feature_to_state():
    state = 0

    # Ghost
    if(features[0] == 1):
        state += 1
    if(features[1] == 1):
        state += 2
    if(features[2] == 1):
        state += 3
    if(features[3] == 1):
        state += 4
    
    # Vulnerable ghost
    if(features[4] == 1):
        state += 5
    if(features[5] == 1):
        state += 10
    if(features[6] == 1):
        state += 15
    if(features[7] == 1):
        state += 20
    
    # Pellet
    if(features[8] == 1):
        state += 25
    if(features[9] == 1):
        state += 50
    if(features[10] == 1):
        state += 75
    if(features[11] == 1):
        state += 100
   
    # Power pellet
    if(features[12] == 1):
        state += 125
    if(features[13] == 1):
        state += 250
    if(features[14] == 1):
        state += 375
    if(features[15] == 1):
        state += 500

    return state

# compare two following actions, return true if directions are opposite and false otherwise
#Good programming etc.
def opposite_directions(dir1, dir2):
    if dir1 == 'RIGHT' and dir2 == 'LEFT' or dir1 == 'LEFT' and dir2 == 'RIGHT':
        return True
    if dir1 == 'UP' and dir2 == "DOWN" or dir1 == "DOWN" and dir2 == "UP":
        return True

    return False

# compute reward from the feature vector in the current state
# R(s,a,s')     s -a-> s' => Q
# s : previousState
# a : previousAction
# s' : currentState
def get_reward():
    global currentPelletCount, hitByGhost, ateGhost#, atePowerPellet

    reward = 0
   
    # Lost a life
    if hitByGhost == 1:
        reward -= 10
        hitByGhost = 0
        # print('pacman lost a life')

    # Ate a pellet
    if currentPelletCount > thisLevel.pellets:
        reward += 3
        # print('pacman ate a pellet')
    currentPelletCount = thisLevel.pellets

    # Ate a ghost
    if ateGhost == 1:
        reward += 0
        ateGhost = 0
        # print('pacman ate a ghost')
    
    # Ate a power pellet
    if thisLevel.atePowerPellet == 1:
        reward += 3
        thisLevel.atePowerPellet = 0
        # print('pacman ate a powerPellet')  
    
 #   for i in range(8,12):   
 #       if oldFeatures[i] == 1 and ACTIONS[i%8] == previousAction:
 #           reward += 2
            # print('Pacman is going for the pellet!')

 #   for i in range(0,4):   
 #       if oldFeatures[i] == 1 and ACTIONS[i] == previousAction:
 #           reward -= 5
            # print('Pacman is going for the ghost!')

    return reward

# compute transition reward to help choose best possible action
def transition_reward(action):
    global returnCounter
    reward = 0

    # action will cause Pacman to return to previous position
    # previousAction = action that lead to this state
    # action = possibleAction
    if opposite_directions(previousAction, action):
         returnCounter += 1
         # print('Pacman evalalutes the ping pong!')
         reward -= 2 * returnCounter
    else: 
         returnCounter = 0

    # Went towards food pellet
    # food pellet: 8 = RIGHT, 9 = LEFT, 10 = DOWN, 11 = UP
    for i in range(8,12):   
        if features[i] == 1 and ACTIONS[i%8] == action:
            reward += 1
            #print('Pacman is going for the pellet!')

    # # Went towards power pellet
    # # power pellet: 12 = RIGHT, 13 = LEFT, 14 = DOWN, 15 = UP
    # for i in range(12,16):   
    #     if features[i] == 1 and ACTIONS[i%12] == action:
    #         reward += 1
    #         #print('Pacman is going for the power pellet!')

    #thisLevel.powerPelletBlinkTimer < tooShortToKill && distance to nearest ghost is too short
    return reward
    


#####################   Q-TABLE HANDLING    #####################
#           UPDATE VARIABLE + LOAD FROM & SAVE TO FILE          #
#################################################################
# load npy-file with q-table to variable Q_table (states x actions)
def load_qtable_from_file():
    global Q_table
    loaded_qtable = np.load("q_table.npy")
    assert loaded_qtable.shape == Q_table.shape, "Q-table sizes do not agree"
    Q_table = loaded_qtable

# update npy-file with current content of variable Q_table
def save_qtable_to_file():
    np.save("q_table.npy", Q_table, allow_pickle=True)
    #np.save(r"E:\Skola\AI\TNM095---project\qtable2.npy", Q_table, allow_pickle=False)

def print_qtable():
    # global Q_table
    np.set_printoptions(threshold=np.inf)
    print(Q_table)



#####################   ACTIONS HANDLING    #####################
#  TRANSLATE STRING TO COORDINATES & GET POSSIBLE/BEST ACTIONS  #
#################################################################
# get the actions from all actions that are possible
# that do not cause pacman to walk into walls
def get_possible_actions():
    possible_actions = []
    for action in ACTIONS:
        dirX, dirY = translateAction(action)  # RIGHT -> [x,y]
        if not thisLevel.CheckIfHitWall((player.x + dirX, player.y + dirY), (player.nearestRow, player.nearestCol)):
            possible_actions.append( actionToInt(action) )

            # Pacman should not be able to walk through ghost door
            row = int(round(player.y + dirY))     
            col = int(round(player.x + dirX))
            if thisLevel.GetMapTile((row, col)) != tileID['ghost-door']:
                possible_actions.append( actionToInt(action) )
       
    # Pacman in tunnel
    if len(possible_actions) == 4:
        if thisLevel.doorH == 1:
            thisLevel.doorH = 0
            possible_actions = []
            possible_actions.append(0)
            possible_actions.append(1)
        elif thisLevel.doorV == 1:
            thisLevel.doorV = 0
            possible_actions = []
            possible_actions.append(2)
            possible_actions.append(3)
        
    return possible_actions

# make string actions grid transformation coordinates
# pacman speed = 3
def translateAction(action):
    if action == 'RIGHT':
        return 3, 0
    if action == 'LEFT':
        return -3, 0
    if action == 'DOWN':
        return 0, 3
    if action == 'UP':
        return 0, -3

# convert string action to an int
def actionToInt(action):
    if action == 'RIGHT':
        return 0
    if action == 'LEFT':
        return 1
    if action == 'DOWN':
        return 2
    if action == 'UP':
        return 3

# convert char actions to ints
def translateChar(char):
    if char == 'R':
        return 0
    if char == 'L':
        return 1
    if char == 'D':
        return 2
    if char == 'U':
        return 3

# get the action among the possible actions 
# that maximizes the reward function
# possibleActions = list of action indices
def get_best_action(possibleActions, currentState):
    maxVal = -np.inf
    bestAction = 5
    
    # get Q-value for each action in current state 
    qvalues = [q_val for q_val in Q_table[currentState]]

    # get possible action with highest Q-value + transition reward
    for action in possibleActions:
        transVal =  transition_reward(ACTIONS[action])
        # print('possibleAction: ', ACTIONS[action],  ' transVal: ', transVal)

        if (qvalues[action] + transVal) > maxVal:
            bestAction = action
            maxVal = qvalues[action] + transVal
        
    # print('Best action: ', ACTIONS[bestAction], '\n')
    return ACTIONS[bestAction]  # return action as string
    

# update Q_table
# Qnew(s,a) <- Q(s,a) + alpha( R(s,a,s') + gamma*max(Q(s',a)) - Q(s,a))     # wikipedia
# => Qnew(s,a) <- (1-alpha)Q(s,a) + alpha*sample
# s = previous state
# a = previous action
# alpha = learning rate

# sample = R(s,a,s') + gamma*max( Q(s',a') )      
# s' = current state
# gamma = discount factor
# a'= action that maximizes Q-value for new state (s')
#------------------------------------------------------
def update_qtable(previousAction, previousState, currentState):
    
    # get old/current Q-value to be updated
    old_q = Q_table[previousState][actionToInt(previousAction)]     # 0: 'RIGHT', 1: 'LEFT', 2: 'DOWN', 3: 'UP'

    # get the estimate of optimal future value
    maxQ = np.max([q_val for q_val in Q_table[currentState]])

    # compute new Q-value with Bellman eq.
    new_q = old_q + LEARNING_RATE * (get_reward() + DISCOUNT * maxQ - old_q)
    Q_table[previousState][actionToInt(previousAction)] = new_q

    # save updated version
    save_qtable_to_file()


#####################   MOVE AGENT IN GUI    ####################
#       RETURNS RIGHT, LEFT, DOWN or UP TO GAME LOGIC           #
#################################################################
def aiMove():
    global prepreAction, previousAction, previousState, EXPLORATION_RATE, oldFeatures

    # print('\naimove')
   # s -a-> s' => Q
   # s : previousState
   # s' : currentState
    oldFeatures = copy.deepcopy(features)

    # update features and get the current state
    calculate_features()
    currentState = feature_to_state()
    # print('currentState: ', currentState)
    # print('Feature[0] = ' , features[0])
    # print('Feature[1] = ' , features[1])
    # print('Feature[2] = ' , features[2])
    # print('Feature[3] = ' , features[3])
    # print('Feature[4] = ' , features[4])
    # print('Feature[5] = ' , features[5])
    # print('Feature[6] = ' , features[6])
    # print('Feature[7] = ' , features[7])
    # print('Feature[8] = ' , features[8])
    # print('Feature[9] = ' , features[9])
    # print('Feature[10] = ' , features[10])
    # print('Feature[11] = ' , features[11])
    # print('Feature[12] = ' , features[12])
    # print('Feature[13] = ' , features[13])
    # print('Feature[14] = ' , features[14])
    # print('Feature[15] = ' , features[15])
  
    # update Q-value for s -a-> s'
    update_qtable(previousAction, previousState, currentState)
    
    # get possible actions for current state
    possibleActions = get_possible_actions()

    # exploration rate - pick random or best action
    if (random.uniform(0, 1) < EXPLORATION_RATE):
        action = ACTIONS[random.choice(possibleActions)]
        # print('random move')
    else:
        action = get_best_action(possibleActions, currentState)    # Q(a', s')

    previousState = currentState
    prepreAction = previousAction
    previousAction = action

    # start game automatically - for training
    if thisGame.mode == 3:
        return 'ENTER'

    # print('action from aimove: ', action)
    return action


# game "mode" variable
# 0 = ready to level start
# 1 = normal
# 2 = hit ghost
# 3 = game over
# 4 = wait to start
# 5 = wait after eating ghost
# 6 = wait after finishing level
# 7 = flashing maze after finishing level
# 8 = extra pacman, small ghost mode
# 9 = changed ghost to glasses
# 10 = blank screen before changing levels

# load previous Q-table
# load_qtable_from_file()
print_qtable()

# initAgent - creating dummy action and first state
if previousState < 0:
    previousAction = 'UP' # dummy action
    calculate_features()
    previousState = feature_to_state()


#####################   MAIN GAME LOOP    ###################
#                                                           #
#############################################################
while deaths < 1000:

    CheckIfCloseButton(pygame.event.get())
    if thisGame.mode == 0:
        # ready to level start
        thisGame.modeTimer += 1

        if aiTraining or thisGame.modeTimer == 150: #Change this to 150 for a brief pause before game
            thisGame.SetMode(1)

    if thisGame.mode == 1:
        # normal gameplay mode
        CheckInputs(aiMove())
        thisGame.modeTimer += 1000

        player.Move()
        for i in range(0, 4, 1):
            ghosts[i].Move()
        thisFruit.Move()

    elif thisGame.mode == 2:
        # waiting after getting hit by a ghost
        thisGame.modeTimer += 1
        hitByGhost = 1
      
        if aiTraining or thisGame.modeTimer == 60: #Change to 60 for longer pause
            thisLevel.Restart()
            thisGame.lives -= 1
         
            if thisGame.lives == -1:
                deaths += 1
                # exp. decreasing exploration rate
                if (EXPLORATION_RATE -0.1) > 0 and deaths > 0 and (deaths%100) == 0:
                    EXPLORATION_RATE -= 0.1
                    print('decreased exploration rate: ' , EXPLORATION_RATE)
        
                # write game data to file 
                file = open('lisas_run_data.txt','a') 
                file.write('-----------------------------\n')
                file.write('---------- death ' + str(deaths) + ' ----------\n')
                file.write('-----------------------------\n')
                file.write('score: ' + str(thisGame.score) + '\n')
                file.write('remaining pellets: ' + str(thisLevel.pellets) + '/140 \n')
                file.write('Exploration rate: ' + str(EXPLORATION_RATE) + '\n')
                file.close()

                #thisGame.updatehiscores(thisGame.score)
                thisGame.SetMode(3)

                #thisGame.drawmidgamehiscores()
             #else:
                #thisGame.SetMode(4)

    elif thisGame.mode == 3:
        # game over
        #thisGame.start
        CheckInputs(aiMove())

    elif thisGame.mode == 4:
        # waiting to start
        thisGame.modeTimer += 1

        if aiTraining or thisGame.modeTimer == 60: #change to 60
            thisGame.SetMode(1)
            player.velX = player.speed

    elif thisGame.mode == 5:
        # brief pause after munching a vulnerable ghost
        thisGame.modeTimer += 1
        ateGhost = 1

        if aiTraining or thisGame.modeTimer == 20:    #change to 20
            thisGame.SetMode(8)

    elif thisGame.mode == 6:
        # pause after eating all the pellets
        thisGame.modeTimer += 1
        
        file = open('lisas_run_data.txt','a') 
        file.write('-----------------------------\n')
        file.write('YOU WON LEVEL 1!! \n' )
        file.write('score: ' + str(thisGame.score) + '\n')
        file.write('remaining pellets: ' + str(thisLevel.pellets) + '/140 \n')
        file.write('Exploration rate: ' + str(EXPLORATION_RATE) + '\n')
        file.write('-----------------------------\n')
        file.close()
        
        # don't load next level or shit
        break
        
        if thisGame.modeTimer == 1:
            thisGame.SetMode(7)
            oldEdgeLightColor = thisLevel.edgeLightColor
            oldEdgeShadowColor = thisLevel.edgeShadowColor
            oldFillColor = thisLevel.fillColor

    elif thisGame.mode == 7:
        # flashing maze after finishing level
        thisGame.modeTimer += 1

        whiteSet = [10, 30, 50, 70]
        normalSet = [20, 40, 60, 80]

        if not whiteSet.count(thisGame.modeTimer) == 0:
            # member of white set
            thisLevel.edgeLightColor = (255, 255, 254, 255)
            thisLevel.edgeShadowColor = (255, 255, 254, 255)
            thisLevel.fillColor = (0, 0, 0, 255)
            GetCrossRef()
        elif not normalSet.count(thisGame.modeTimer) == 0:
            # member of normal set
            thisLevel.edgeLightColor = oldEdgeLightColor
            thisLevel.edgeShadowColor = oldEdgeShadowColor
            thisLevel.fillColor = oldFillColor
            GetCrossRef()
        elif thisGame.modeTimer == 100:
            thisGame.SetMode(10)

    elif thisGame.mode == 8:
        CheckInputs(aiMove())
        ghostState = 1
        thisGame.modeTimer += 1

        player.Move()

        for i in range(0, 4, 1):
            ghosts[i].Move()

        for i in range(0, 4, 1):
            if ghosts[i].state == 3:
                ghostState = 3
                break
            elif ghosts[i].state == 2:
                ghostState = 2

        if thisLevel.pellets == 0:
            # WON THE LEVEL
            thisGame.SetMode(6)
        elif ghostState == 1:   # Ghost are normal
            thisGame.SetMode(1)
        elif ghostState == 2:   # Ghosts are vulnerable
            thisGame.SetMode(9)

        thisFruit.Move()

    elif thisGame.mode == 9:
        # atePowerPellet = 1      # This will be 1 for as long as the ghosts are vulnerable
        CheckInputs(aiMove())
        thisGame.modeTimer += 1

        player.Move()
        for i in range(0, 4, 1):
            ghosts[i].Move()
        thisFruit.Move()

    elif thisGame.mode == 10:
        # blank screen before changing levels
        thisGame.modeTimer += 1
        if thisGame.modeTimer == 10:
            thisGame.SetNextLevel()

    elif thisGame.mode == 11:
        # flashing maze after finishing level
        thisGame.modeTimer += 1

        whiteSet = [10, 30, 50, 70]
        normalSet = [20, 40, 60, 80]

        if not whiteSet.count(thisGame.modeTimer) == 0:
            # member of white set
            thisLevel.edgeLightColor = (255, 255, 254, 255)
            thisLevel.edgeShadowColor = (255, 255, 254, 255)
            thisLevel.fillColor = (0, 0, 0, 255)
            GetCrossRef()
        elif not normalSet.count(thisGame.modeTimer) == 0:
            # member of normal set
            thisLevel.edgeLightColor = oldEdgeLightColor
            thisLevel.edgeShadowColor = oldEdgeShadowColor
            thisLevel.fillColor = oldFillColor
            GetCrossRef()
        elif thisGame.modeTimer == 100:
            thisGame.modeTimer = 1

    thisGame.SmartMoveScreen()

    screen.blit(img_Background, (0, 0))

    if not thisGame.mode == 10:
        thisLevel.DrawMap()

        if thisGame.fruitScoreTimer > 0:
            if thisGame.modeTimer % 2 == 0:
                thisGame.DrawNumber(2500, (
                    thisFruit.x - thisGame.screenPixelPos[0] - 16, thisFruit.y - thisGame.screenPixelPos[1] + 4))

        for i in range(0, 4, 1):
            ghosts[i].Draw()
        thisFruit.Draw()
        player.Draw()

        if thisGame.mode == 3:
            screen.blit(thisGame.imHiscores, (HS_XOFFSET, HS_YOFFSET))

    if thisGame.mode == 5:
        thisGame.DrawNumber(thisGame.ghostValue / 2,
                            (player.x - thisGame.screenPixelPos[0] - 4, player.y - thisGame.screenPixelPos[1] + 6))

    thisGame.DrawScore()

    pygame.display.update()
    del rect_list[:]


    clock.tick(40.0 + aiTraining * 1)

 