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
NUM_FEATURES = 2    # states = 2^NUM_FEATURES = 
NUM_ACTIONS = 4     # RIGHT, LEFT, DOWN or UP 
ACTIONS = ['RIGHT', 'LEFT', 'DOWN', 'UP'] 
LEARNING_RATE = 0.5 # alpha [0,1]
EXPLORATION_RATE = 0.1
DISCOUNT = 0.8 # gamma [0,1]

Q_table = np.zeros([2**NUM_FEATURES, NUM_ACTIONS])
features = np.zeros([NUM_FEATURES]).astype(int) # binary


# global variables
reward = 0
previousState = -1
previousAction = 'LEFT'
currentPelletCount = 140
hitByGhost = 0
ateGhost = 0


#####################   FEATURE DEFINITIONS    ######################
#           COMPUTE GHOST/PELLET/ENERGIZER DISTANCE                 #
#####################################################################
# compute distance to closest ghost in given state
# return 1 if smaller than threshold and 0 otherwise
#   1 = normal
#   2 = vulnerable
#   3 = spectacles
def ghost_distance(state = 1):
    threshold = 150

    for i in range(0, 4, 1):
        if ghosts[i].state == state:
            distance = abs(player.x - ghosts[i].x) + abs(player.y - ghosts[i].y)
            if distance < threshold:
                return 1
    return 0

# TODO: we are here        
# compute distance to closest food pellet
# return 1 if closer than threshold and 0 otherwise
def pellet_distance():

    return 0

#####################   FEATURE  & REWARD COMPUTATIONS    ###################
#       BINARY RESULTS FOR EACH FEATURE & POS/NEG REWARDS FOR ACTIONS       #
#############################################################################
#
# features[0]: will be 1 if a normal ghost is close
# features[1]: will be 1 if vulnerable ghost is close
# features[2]: will be 1 if food pellet is close
# features[3]: will be 1 if time of closest vulnerable ghost is short
# features[4]: will be 1 if energizer is close

def calculate_features(pos = []):
    idx = 0   
    
    # normal ghost distance
    features[idx] = ghost_distance()
    idx += 1

    # vulnerable ghost distance (state = 2)
    features[idx] = ghost_distance(2)
    idx += 1

    # food pellet close!
    # features[idx] = pellet_distance()
    # idx += 1

def feature_to_state():
    state = 0

    if(features[0] == 1):
        state += 1
    if(features[1] == 1):
        state += 2

    return state

# compute reward from the feature vector in the current state
def get_reward():
    global currentPelletCount, hitByGhost, ateGhost
    
    rwd = 0
   
    # Lost a life
    if hitByGhost == 1:
        rwd -= 10
        hitByGhost = 0
        print('pacman lost a life')

    # Eating a pellet
    if currentPelletCount > thisLevel.pellets:
        rwd += 1
        print('pacman ate a pellet')

    currentPelletCount = thisLevel.pellets
         
    # thisLevel.powerPelletBlinkTimer < tooShortToKill && distance to nearest ghost decreases
    # rwd -= 1
    
    # Eating a ghost
    if ateGhost == 1:
        rwd += 10
        ateGhost = 0
        print('pacman ate a ghost')
        
    
    # Time step: 
    # rwd -= 1

    # Revisit previous/empty positions:
    # prevprev & previousAction are opposite
    
    # for testing
    #rwd = 5
    if rwd != 0:
        print('reward: ', rwd)
    return rwd
    

#####################   Q-TABLE HANDLING    #####################
#           UPDATE VARIABLE + LOAD FROM & SAVE TO FILE          #
#################################################################
# load npy-file with q-table to variable Q_table (states x actions)
def load_qtable_from_file():
    global Q_table
    loaded_qtable = np.load("qtable.npy")
    assert loaded_qtable.shape == Q_table.shape, "Q-table sizes do not agree"
    Q_table = loaded_qtable

# update npy-file with current content of variable Q_table
def save_qtable_to_file():
    np.save("qtable", Q_table)



#####################   ACTIONS HANDLING    #####################
#  TRANSLATE STRING TO COORDINATES & GET POSSIBLE/BEST ACTIONS  #
#################################################################
# get the actions from all actions that are possible
# that do not cause pacman to walk into walls
def get_possible_actions(ACTIONS):
    possible_actions = []
    for action in ACTIONS:
        move = translateAction(action)  # RIGHT -> [x,y]
        if not thisLevel.CheckIfHitWall((player.x + move[0], player.y + move[1]), (player.nearestRow, player.nearestCol)):
            possible_actions.append( actionToInt(action) )

    return possible_actions

# make string actions grid transformation coordinates
# pacman speed = 3
def translateAction(action):
    if action == 'RIGHT':
        return [3,0]
    if action == 'LEFT':
        return [-3,0]
    if action == 'DOWN':
        return [0,3]
    if action == 'UP':
        return [0,-3]

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

    # get possible action with highest Q-value
    for action in possibleActions:
        if qvalues[action] > maxVal:
            bestAction = action
            maxVal = qvalues[action]

    print('possible action with highest Q-value: ', ACTIONS[bestAction], ' , value: ', maxVal)
    return ACTIONS[bestAction]  # return action as string
    



# update Q_table
def update_qtable(previousAction, previousState, currentState):
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

    # np.set_printoptions(precision=3)
    # print(Q_table)

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
    global previousAction, previousState
    #print('normal ghost', features[0])
    #print('vulnerable ghost', features[1])
    #print('ghost to the right: ', check_close_ghost('RIGHT'))
    #print('ghost to the left: ', check_close_ghost('LEFT'))
    #print('ghost to the up: ', check_close_ghost('UP'))
    #print('ghost to the down: ', check_close_ghost('DOWN'))
    #print('player pos: ', player.x, player.y)
    #print('ghost[0] pos: ', ghosts[0].x, ghosts[0].y)


   # s -a-> s' => Q
   # s : previousState
   # s' : currentState

    # update features and get the current state
    calculate_features()
    currentState = feature_to_state()
  
    # update Q-value for s -a-> s'
    update_qtable(previousAction, previousState, currentState)

    # get possible actions for current state
    possibleActions = get_possible_actions(ACTIONS)

    # exploration rate - pick random or best action
    if (random.uniform(0, 1) < EXPLORATION_RATE):
        action = random.choice(possibleActions)
        print('random move')
    else:
        action = get_best_action(possibleActions, currentState)    # Q(a', s')
    
    previousState = currentState
    previousAction = action
  
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

aiTraining = 1
deaths = 0

# load previous Q-table
load_qtable_from_file()
print(Q_table)

# initAgent - creating dummy action and first state
if previousState < 0:
    previousAction = 'LEFT' # dummy action
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

                thisGame.updatehiscores(thisGame.score)
                thisGame.SetMode(3)
                thisGame.drawmidgamehiscores()
            else:
                thisGame.SetMode(4)

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

    clock.tick(40 + aiTraining * 1)
 