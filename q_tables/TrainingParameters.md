
Training rounds: 	10000

Exploration rate: 	1.0 -> 0.0 over 4500 rounds
Learning rate: 		0.0005
Discount factor: 	0.95
GhostClose:		6

TRAINING 01 
Hit by ghost: 		-50
Eat pellet: 		10
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	2
Go towards ghost: 	-5
Timestep		0
Cancelled after 5235 iterations.
Never walks a straight line!! 
Keeps reversing one step and then keep moving forward like he has some tics. 
Gets stuck in corners. 



TRAINING 02 
Hit by ghost: 		-50
Eat pellet: 		10
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	0	"No cheating pellet"
Go towards ghost: 	0	"No cheating ghost"
Timestep		0
Cancelled after 7000 iterations.
Similar to 05 in behaviour. 



TRAINING 03 
Hit by ghost: 		-30	Smaller ghost impact
Eat pellet: 		10
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	2
Go towards ghost: 	-5
Timestep		0
Cancelled after 7554 iterations.
Goes back n forth in corners/intersections. 
Especially when he enters a tile where he has 3 road options it seems. 
Gets out of behaviour sometimes when ghost approaches



TRAINING 04 
Hit by ghost: 		-30	Smaller ghost impact
Eat pellet: 		5	Smaller pellet impact
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	2
Go towards ghost: 	-5
Timestep		0
Cancelled after 6230 iterations.
Same problem as 03. 


TRAINING 05 
Hit by ghost: 		-30	Smaller ghost impact
Eat pellet: 		5	Smaller pellet impact
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	0	"No cheating pellet"
Go towards ghost: 	0	"No cheating ghost"
Timestep		0	
Cancelled after 7300 iterations. 
Best so far. Will walk toward ghost if pellet is in same direction and he is already walking that direction
because reversing will yield a tiny bit too much punishment. Improvement ideas below. 

TRAINING 06 -
Hit by ghost: 		-50
Eat pellet: 		10
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	2
Go towards ghost: 	-5
Timestep		-1	Timestep punishment
Cancelled after 6060 iterations.
Not much different from 01.






WHEN TRAINING CRASHES
Backup q_table
uncomment load qtable
Change exploration rate
Change deaths
save the agent.py document

WHEN TRAINING IS DONE
Save backup of q_table and rename the backup to q_table_0x.npy 


WHEN CREATING NEW TRAINING SESSION
Set deaths to 0
Set learning, exploration and discount
Comment out load_qtable
Change the prints to ('TRAINING0x deaths: ', deaths) and the print to the textfile below it
Set a high tick!

During testing, set exploration and learning rate to 0 


OBSERVATIONS DURING TRAINING
02 and 05 dies at a faster rate, indicating they do not escape ghosts as well at the moment. They have the "no cheating"!
After 3000 iterations 02 does not think it is worth it to turn to the right to eat a pellet, if he previously was going to the left. 
Maybe he needs bigger reward for eating pellet.
At 2100 iterations 01 ping pongs a lot unless there is a ghost nearby to scare him in one directon. 
02 and 05 walk into ghosts and get the recursion error more often. Maybe because they don't have the "cheat" and are less scared of ghosts.

At 2892 01 ping pongs like hell in intersections and corners. 
Maybe he gets too much punishment from ghost, and therefore reverse back and forth because that still costs less than getting hit by ghost?
03 does the same thing even tho he has a bit smaller ghost impact. 04 and 06 suffers from same behaviour. 
05 seems to do longer ping pongs instead, but also doesn't really seem to travel the map as much. 

High ghost impact makes pacman afraid to run past a ghost even though he would be able to cross the T-intersection before the ghost, 
making him turn back and become trapped in corner because ghosts also come from other way. 

Some pacmans really starts to freak out and ping pong in the same space when there are ghosts nearby. 
When the ghosts turn blue, they stop ping pong. Or at least there is usually a clear change in pacmans behaviour after 
he eats a power pill.

At 4500ish, 04 can not handle intersections, unless chased by ghost. 03 displays same behaviour. 
This Does not improve at 5500 (6500 for 03)

At around 4-5k deaths, the difference between 02 and 05 is that 02 does more short ping pongs (2-3 tiles) and 05 does longer sweeps running back n forth in hallways.
05 has lower pellet reward than 02. None of them has the "cheats". 05 consistently misses pellets that are not in straight lines. But he almost never ping pongs short!
High pellet reward maybe leads to easier to turn but higher risk of high freq ping pong. 

05 does not seem to freak out about ghosts (at 6k) and can walk towards them, sometimes too much, even walk straight into them. 
This makes it so he does not get trapped in corners. 

When 05 has pellet to the right and prev action right and ghost to the right, he rather goes into the ghost than turn, 
because of the reverse punishment, and because he has experienced so many times that "walking towards ghost direction doesnt punish me"
since he has "no cheat". Ideas for solution: Make ghost dist threshold smaller, or increase impact of ghost. 



TRY NEXT:

What if you give 05 a punishment for each time step? Will this discourage the behaviour of the long sweeps?

TRAINING 05.1 
Hit by ghost: 		-50	
Eat pellet: 		5	Smaller pellet impact
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	0	"No cheating pellet"
Go towards ghost: 	0	"No cheating ghost"
Timestep		0	

TRAINING 05.2
Hit by ghost: 		-30	Smaller ghost impact
Eat pellet: 		10	
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	0	"No cheating pellet"
Go towards ghost: 	0	"No cheating ghost"
Timestep		0	

TRAINING 05.3 
Hit by ghost: 		-40	Medium ghost impact	
Eat pellet: 		5	Smaller pellet impact
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	0	"No cheating pellet"
Go towards ghost: 	0	"No cheating ghost"
Timestep		0	

TRAINING 05.4 
Hit by ghost: 		-30	Smaller ghost impact
Eat pellet: 		5	Smaller pellet impact
Reverse in path: 	-2
Eat ppellet:		5
Go towards pellet:	0	"No cheating pellet"
Go towards ghost: 	0	"No cheating ghost"
Timestep		0	
Ghost distance < 4 tiles!




TRAINING 07 - Default but more reverse punishment
Hit by ghost: 		-50		
Eat pellet: 		10
Reverse in path: 	-13		To match the values in the report. Because he will get 16+10 for each pellet if keeping a straight line.
Eat ppellet:		5
Go towards pellet:	2
Go towards ghost: 	-5

GhostClose:		6