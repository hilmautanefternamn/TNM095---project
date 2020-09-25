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

# constants
NUM_FEATURES = 2    # states = 2^NUM_FEATURES = 
NUM_ACTIONS = 4     # RIGHT, LEFT, DOWN or UP 
actions = ['RIGHT', 'LEFT', 'DOWN', 'UP'] 
LEARNING_RATE = 0.5
EXPLORATION_RATE = 1
GAMMA = 0.8

Q_table = np.zeros([2**NUM_FEATURES, NUM_ACTIONS])
features = np.zeros([NUM_FEATURES]).astype(int) # binary
print("q_table size:", Q_table.shape)


reward = 0
nextState = -1
action = ''

# DUMMY INIT
def initAgent():
    calculate_features()
    nextState = feature_to_state()
    action = 'LEFT' # dummy action


if nextState < 0:
    initAgent()

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
                    

#####################   FEATURE  & REWARD COMPUTATIONS    ###################
#       BINARY RESULTS FOR EACH FEATURE & POS/NEG REWARDS FOR ACTIONS       #
#############################################################################
#
# features[0]: will be 1 if a normal ghost is close
# features[1]: will be 1 if vulnerable ghost is close
# features[2]: will be 1 if time of closest vulnerable ghost is short
# features[3]: will be 1 if food pellet is close
# features[4]: will be 1 if energizer is close

def calculate_features(pos = []):
    idx = 0   
    
    # normal ghost distance
    features[idx] = ghost_distance()
    idx += 1

    # vulnerable ghost distance
    features[idx] = ghost_distance(2)
    idx += 1

def feature_to_state():
    state = 0

    if(features[0] == 1):
        state += 1
    if(features[1] == 1):
        state += 2

    return state


def get_reward(action):
    
    # Eating a pellet:
    # Killing a ghost:
    # Losing life/Dying:
    # Time step: 
    # Revisit previous/empty positions:
    reward = 0
    

#####################   Q-TABLE HANDLING    #####################
#           UPDATE VARIABLE + LOAD FROM & SAVE TO FILE          #
#################################################################
# load npy-file with q-table to variable Q_table (states x actions)
def load_qtable_from_file():
    loaded_qtable = np.load("qtable.npy")
    assert loaded_qtable.shape == Q_table.shape, "Q-table sizes do not agree"
    Q_table = loaded_qtable

# update npy-file with current content of variable Q_table
def save_qtable_to_file():
    np.save("qtable", Q_table)

# update Q_table
# currentState: result of last performed action -> previousAction
# nextState: result of the new action to be performed -> action
def update_qtable(previousAction, currentState, nextState):
    qState = Q_table[currentState]
    new_qState = Q_table[nextState]
    
    maxQ = np.max(new_qState)

    
    # Q(s,a) <- (1-alpha)Q(s,a) + alpha*sample
    # s = previous state
    # a = previous action
    # alpha = learning rate

    # sample = R(s,a,s') + gamma*max( Q(s',a') )      
    # s' = new state
    # gamma = discount factor
    # a'= action that maximizes Q-value for new state (s')
    

    # save updated version
    save_qtable_to_file()


#####################   ACTIONS HANDLING    #####################
#  TRANSLATE STRING TO COORDINATES & GET POSSIBLE/BEST ACTIONS  #
#################################################################
# get the actions from all actions that are possible
# that do not cause pacman to walk into walls
def get_possible_actions(actions):
    possible_actions = []
    for action in actions:
        move = translateAction(action)  # RIGHT -> [x,y]
        if not thisLevel.CheckIfHitWall((player.x + move[0], player.y + move[1]), (player.nearestRow, player.nearestCol)):
            possible_actions.append(action)

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

# get the action among the possible actions 
# that maximizes the reward function
def get_best_action(actions):
    maxReward = 0
    bestAction = ''

    # get Q-value for each feature in current state
    qvalues = []

    for value in qvalues:

    # compute reward change for each move
        for action in actions:
            rwd = get_reward(action)
            # found better move
            if  rwd > maxReward:
                maxReward = rwd
                bestAction = action

    return bestAction


#####################   MOVE AGENT IN GUI    ####################
#       RETURNS RIGHT, LEFT, DOWN or UP TO GAME LOGIC           #
#################################################################
def aiMove():
    #print('normal ghost', features[0])
    #print('vulnerable ghost', features[1])
    #print('ghost to the right: ', check_close_ghost('RIGHT'))
    #print('ghost to the left: ', check_close_ghost('LEFT'))
    #print('ghost to the up: ', check_close_ghost('UP'))
    #print('ghost to the down: ', check_close_ghost('DOWN'))
    #print('player pos: ', player.x, player.y)
    #print('ghost[0] pos: ', ghosts[0].x, ghosts[0].y)

    currentState = nextState

    # update features and get newState
    calculate_features()
    nextState = feature_to_state()
   
    # save previous action before choosing a new one
    previousAction = action

    # get possible actions for current position
    possible_actions = get_possible_actions(actions)

    # exploration rate - pick random or best action
    if (random.uniform(0, 1) < EXPLORATION_RATE):
        action = random.choice(possible_actions)
    else:
        action = get_best_action(possible_actions, nextState)

    # currentState: result of last performed action -> previousAction
    # nextState: result of the new action to be performed -> action
    update_qtable(previousAction, currentState, nextState)
    
  
    if thisGame.mode == 3:
        return 'ENTER'
   
    return # action


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

    clock.tick(10 + aiTraining * 0)
 