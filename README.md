# Playing-blackjack-with-Reinforcement-Learning
UCLA PYSCH196B Neural Network Class Project 

Implements reinforcement learning agents to play blackjack with their memory information.
1. Agents are assigned different memory size which limits the maximum number of past cards they can remember
2. Agents used these their memory information along with current game information to play the game

Two different model structures are developed to train these agents:
1. DQN only
2. DQN + Belief Network

Belief Network
We designed this seperate neural network specifically to let the agent process their memory information and develpe a belief over the remaining cards distribution, which is then used as a part of the input for DQN. 
