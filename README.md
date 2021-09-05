# interactive-AI-game
## Welcome to this project!

This is a prototype of a game which lets the player control a predator, whose objective is to devour prey agents. The agents learn to avoid the player from experience in real time. My interest for this subject arose from not previously having seen a project of this sort. Machine learning has avidly been applied for games such as Go, Tetris, Super Mario etc. However, I haven't yet witnessed it as an actual mechanism in a video game. 

I'll proceed to give a brief explanation of the construction of the game.

## Neural network
### LSTM

The agents are controlled by an LSTM, which is fed time series consisting of:
1. distances to the predator,
2. angles in relation to the predator direction.

The LSTM gives predictions of which future states are the most beneficial to the agents. The agents can accelerate up, down, left or right. The future states are estimated by allowing the agents to try out all of the possible directions, and the LSTM ranks all of them. The direction with the highest Q-value is chosen for each of the agents.

### Q-learning

Rewards are given to agents depending on how long they survive. An agent who survives below the limit of the generation length is penalised, but with lower penalty if it survives for longer. If the agent survives for the entire generation, it is positively rewarded. These rewards are given for states with a certain interval of frames apart from each other. An $\epsilon$-greedy policy is applied, meaning that agents are allowed to randomly explore, as well as exploit the network. $\epsilon$, which is the fraction of chosen random moves, starts out high and gradually decreases. 

## Some videos of the game

### Early generational behaviour
For $\epsilon\approx 0.9$, we can observe a very random behaviour. The agents have no clear strategy. It is, however, still a viable means of surviving, as it is difficult to predict randomness. 
https://user-images.githubusercontent.com/54723095/132140076-98793407-0979-4083-b2d4-49d02d5e998e.mp4

### Mid generational behaviour
For $\epsilon\approx 0.5$, agents start to get more coordinated. Typically, we can observe agents going along walls away from the predator.
https://user-images.githubusercontent.com/54723095/132140081-00db869e-e8a2-4804-95b7-5ccc924e6b76.mp4

### Late generational behaviour
For low $\epsilon$, the above strategy is even more clearly seen.
https://user-images.githubusercontent.com/54723095/132139895-8fd17b2b-7c4b-4ab4-af55-bed68e555085.mp4

