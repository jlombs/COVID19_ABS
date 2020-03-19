# COVID19
A Python3 agent-based simulator for the 2019 novel coronavirus

## Introduction
This simple simulator is used to illustrate the effects of various social behaviors on the spread and impact of COVID19. 
The project was inspired by https://www.washingtonpost.com/graphics/2020/world/corona-simulator/ which brilliantly illustrates the
same. However, these simulations are a bit limited in that they model iteractions from a kinetic theory billiard-ball perspective.
The virus isn't just spread from direct human-to-human contact, but from surfaces etc. Our simulator was 
designed from the ground-up to study environmental transmission. We use natural human behaviors for local movement, and also 
include other factors like the ability to clean one's local environment. Unlike the existing social distancing models that 
freeze-out the mobile degrees of freedom of the individuals, ours gives the agents varying propensity for making smart choices
to stay away from their neighbors. We also use actual COVID-19 and world population statistics to govern infection data, etc, 
where applicable.

## Dependencies
numpy, scipy, matplotlib

## Execution
The scripts and classes are internally documenting, and the simulate.py script contains a user interface section for playing with
the model. Running that script is all that is necessary to see the simulations.

## Future work
We aim to include elements like re-infection possibilities, as well as hospital capacity and influence on death rates. We
also aim to calibrate the infection rates to map even more directly onto the latest knowledge about COVID-19. This project
is just for fun to gain intuition for how a virus like COVID-19 spreads, and should not be used to make any healthcare decisions.
This being said, clean up and socially distance yourselves!
