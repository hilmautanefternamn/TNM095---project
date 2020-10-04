#  AI-agent who can learn Pacman through feature based Q-learning and reinforcement learning

# FEATURES :
#------------------------------------
# Distance to closest normal ghost: x tiles away
# Distance to closest vulnerable ghost: x tiles away
# Time left of closest vulnerable ghost: 
# Walls in neighbouring location: 1 tile away 
# Distance to closest pellets: x tiles away
# Distance to closest energizer: x tiles away
# (Distance to closest fruit)

# ACTIONS : # RIGHT # LEFT # DOWN # UP

############    Q-TABLE SET-UP  #################
#                           R   L   D   U
#   normal ghost            x   x   x   x
#   vulnerable ghost        x   x   x   x
#   time vulnerable         x   x   x   x
#   pellet                  x   x   x   x
#   energizer               x   x   x   x
#
#################################################

# REWARDS :
#----------------------
# Eating a pellet:
# Killing a ghost:
# Losing life/Dying:
# Time step: 
# Revisit previous/empty positions:
# Walk into wall:

from pacman import *
import random
import numpy as np 

# global constants
NUM_FEATURES = 16    
NUM_STATES = 625        # 2**NUM_FEATURES 
NUM_ACTIONS = 4         # RIGHT, LEFT, DOWN or UP 
ACTIONS = ['RIGHT', 'LEFT', 'DOWN', 'UP'] 
LEARNING_RATE = 0.8     # alpha [0,1]
EXPLORATION_RATE = 0.5  # [0,1]
DISCOUNT = 0.7          # gamma [0,1]

Q_table = np.zeros([NUM_STATES, NUM_ACTIONS])
features = np.zeros([NUM_FEATURES]).astype(int) # binary

# global variables
previousState = -1
previousAction = 'LEFT'
prepreAction = 'LEFT'
currentPelletCount = 140
hitByGhost = 0
ateGhost = 0
towardsPellet = 0
returnCounter = 0
atePowerPellet = 0

aiTraining = 1
deaths = 0


#####################   FEATURE DEFINITIONS    ######################
#           COMPUTE GHOST/PELLET/ENERGIZER DISTANCE                 #
#####################################################################

# Compute distance to the closest ghost in given direction
#   1 = normal
#   2 = vulnerable
#   (3 = spectacles)
def ghost_distance(action, state):
    dirX, dirY = translateAction(action)     # 'RIGHT' -> (3,0)
    posX = player.x + dirX
    posY = player.y + dirY
    minDist = np.inf

    for i in range(0, 4, 1):
        if ghosts[i].state == state:
            distance = abs(posX - ghosts[i].x) + abs(posY - ghosts[i].y)
            if distance < minDist:
                minDist = distance
    return minDist

# get action giving the smallest distance to closest ghost
def closest_ghost_dir(state = 1):
    distance = []
    # find minDist to closest ghost for each action
    for action in ACTIONS:
        distance.append(ghost_distance(action, state))

    # return action giving smallest minDist = worst choice of direction
    idx = distance.index(min(distance))
    return ACTIONS[idx], min(distance)


# compute closest distance to a pellet in given direction
def pellet_distance(action, radius = 1, pelletType = 'pellet'):
    dirX, dirY = translateAction(action)     # 'RIGHT' -> (3,0)
    posX = (player.x + dirX) / TILE_HEIGHT
    posY = (player.y + dirY) / TILE_WIDTH
    minDist = np.inf

    # Loop grid around tile 
    for r in range(-radius, radius + 2):        # we add with 2 cuz he did it in the game
        for c in range(-radius, radius + 2):
            # scale coordinates to tiles
            row = int( posY + r*(TILE_WIDTH / 2) / TILE_WIDTH )
            col = int( posX + c*(TILE_HEIGHT / 2) / TILE_HEIGHT)

            # if tile contains pellet, compute distance 
            if (thisLevel.GetMapTile((row, col)) == tileID[pelletType]):
                # len(path.FindPath((player.nearestRow, player.nearestCol), (row,col))) 
                pelletPath = path.FindPath((player.nearestRow, player.nearestCol), (row,col)) 
                #print(pelletPath)
                #distance = abs(posX - col) + abs(posY - row)
                #if distance < minDist:
                #    minDist = distance
                if pelletPath != False and len(pelletPath) > 0 and len(pelletPath) < minDist:
                    minDist = len(pelletPath)
                    #print('shortest path to ', pelletType, ': ', pelletPath)
    
    return minDist

# get action giving the smallest distance to the closest pellet
# pelletType: 'pellet', 'pellet-power'
def closest_pellet_dir(radius = 1, pelletType = 'pellet'):
    distance = []
    for action in ACTIONS:
        distance.append(pellet_distance(action, radius, pelletType))
    
    idx = distance.index(min(distance))
    return ACTIONS[idx], min(distance)


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
    GHOSTCLOSE = 35 # ~ 5 tiles away TODO: set to 120

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
    # TODO: what happens when blue ghost doesnt exist or disappear?
    for action in ACTIONS:
        if vghostDist < GHOSTCLOSE and action == vghostDirection:
            features[idx] = 1
        else:
            features[idx] = 0
        idx += 1

    # food pellet distance for every action
    pelletDirection, pelletDist = closest_pellet_dir(2)
    for action in ACTIONS:
        if pelletDist < 9999 and action == pelletDirection:
            features[idx] = 1
        else:
            features[idx] = 0
        idx += 1

    # power pellet distance for every action
    ppelletDirection, ppelletDist = closest_pellet_dir(1, 'pellet-power')
    for action in ACTIONS:
        if ppelletDist < 9999 and action == ppelletDirection:
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




    # if(features[0] == 1):
    #     state += 1
    # if(features[1] == 1):
    #     state += 2
    # if(features[2] == 1):
    #     state += 4
    # if(features[3] == 1):
    #     state += 8
    # if(features[4] == 1):
    #     state += 16
    # if(features[5] == 1):
    #     state += 32
    # if(features[6] == 1):
    #     state += 64
    # if(features[7] == 1):
    #     state += 128
    # if(features[8] == 1):
    #     state += 256
    # if(features[9] == 1):
    #     state += 512
    # if(features[10] == 1):
    #     state += 1024
    # if(features[11] == 1):
    #     state += 2048
    # if(features[12] == 1):
    #     state += 4096
    # if(features[13] == 1):
    #     state += 8192
    # if(features[14] == 1):
    #     state += 16384
    # if(features[15] == 1):
    #     state += 32768

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
    global currentPelletCount, hitByGhost, ateGhost, towardsPellet, atePowerPellet
    
    reward = 0
   
    # Lost a life
    if hitByGhost == 1:
        reward -= 15
        hitByGhost = 0
        #print('pacman lost a life')

    # Ate a pellet
    if currentPelletCount > thisLevel.pellets:
        reward += 3
        #print('pacman ate a pellet')
    currentPelletCount = thisLevel.pellets

    # Ate a ghost
    if ateGhost == 1:
        reward += 2
        ateGhost = 0
        #print('pacman ate a ghost')
    
    # Ate a power pellet
    if atePowerPellet == 1:
       reward += 5
       atePowerPellet = 0

    # Went towards pellet
    for i in range(8,12):   # 8 = RIGHT, 9 = LEFT, 10 = DOWN, 11 = UP
        if features[i] == 1 and ACTIONS[i%8] == previousAction:
            reward += 1
            #print('Pacman is going for the pellet!')
        
    #if rwd != 0:
     #   print('reward: ', rwd)

    # reward += transition_reward(prepreAction)

    return reward

# compute transition reward to help choose best possible action
def transition_reward(action):
    global returnCounter
    reward = 0

    # no pellet dir found - keep walking in the same direction
    foundPelletDir = 0
    for i in range(8,12):   # 8 = RIGHT, 9 = LEFT, 10 = DOWN, 11 = UP
        if features[i] == 1:
            foundPelletDir = 1
        
    if foundPelletDir == 0 and previousAction == action: 
        reward += 2
        #print('pacman is searching for a pellet')

    
    # action will cause Pacman to return to previous position
    if opposite_directions(previousAction, action):
        returnCounter += 1
        #print('Pacman does the ping pong!')
        reward -= 2 * returnCounter
    else: 
        returnCounter = 0

    # thisLevel.powerPelletBlinkTimer < tooShortToKill && distance to nearest ghost is too short

    return reward
    

#####################   Q-TABLE HANDLING    #####################
#           UPDATE VARIABLE + LOAD FROM & SAVE TO FILE          #
#################################################################
# load npy-file with q-table to variable Q_table (states x actions)
def load_qtable_from_file():
    global Q_table
    loaded_qtable = np.load("q_table_lisa.npy")
    assert loaded_qtable.shape == Q_table.shape, "Q-table sizes do not agree"
    Q_table = loaded_qtable

# update npy-file with current content of variable Q_table
def save_qtable_to_file():
    np.save("q_table_lisa.npy", Q_table, allow_pickle=True)
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

        # print('pre: ', previousAction, ' , action: ', ACTIONS[action])
        # print('trans: ', transVal)

        if qvalues[action] + transVal > maxVal:
            bestAction = action
            maxVal = qvalues[action]

    #print('possible action with highest Q-value: ', ACTIONS[bestAction], ' , value: ', maxVal)
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
    global prepreAction, previousAction, previousState, deaths

   # s -a-> s' => Q
   # s : previousState
   # a : previousAction
   # s' : currentState

    # update features and get the current state
    calculate_features()
    currentState = feature_to_state()
  
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

    if EXPLORATION_RATE > 0 and deaths%100 == 0:
        EXPLORATION_RATE - 0.1

    
    previousState = currentState
    prepreAction = previousAction
    previousAction = action

    
    # start game automatically - for training
    if thisGame.mode == 3:
        return 'ENTER'
   
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

# KOLLA numpy.savetxt
# spara till leslig textfil

# load previous Q-table
load_qtable_from_file()
# np.set_printoptions(threshold=np.inf)
print_qtable()

# initAgent - creating dummy action and first state
if previousState < 0:
    previousAction = 'UP' # dummy action
    calculate_features()
    previousState = feature_to_state()


#####################   MAIN GAME LOOP    ###################
#                                                           #
#############################################################
while True:

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
                # write game data to file 
                file = open('pacman_run_data.txt','a') 
                file.write('-----------------------------\n')
                file.write('---------- death ' + str(deaths) + ' ----------\n')
                file.write('-----------------------------\n')
                file.write('score: ' + str(thisGame.score) + '\n')
                file.write('remaining pellets: ' + str(thisLevel.pellets) + '/140 \n')
                file.close()
                #L = ["-----------------------------\n", "score: " + str(thisGame.score) + "\n", "remaining pellets: " + str(thisLevel.pellets) + "/140\n"]   
                #file.writelines(L) 

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
        elif ghostState == 1:
            thisGame.SetMode(1)
        elif ghostState == 2:
            thisGame.SetMode(9)

        thisFruit.Move()

    elif thisGame.mode == 9:
        atePowerPellet = 1
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

    clock.tick(40 + aiTraining * 9999999)
 