# Script to execute the simulation

from ABClasses import agent, environment
from matplotlib import pyplot as plt
import numpy as np

# User Interface
##########################################
# Establish the population: see the ABClasses.py documentation for details
populationSize = 250  # Must be >= 10, expect slowdowns for large social distancing on large populations
population = [agent(ageBias=-1, cleanlinessBias=0*.5*np.random.rand(), travelerBias=1.0+2.5*np.random.rand(), socialDistanceBias=0+0*np.random.rand()) for _ in range(populationSize)]

# Establish the simulation environment: see the ABClasses.py documentation for details
landscape = environment(AOE=5, attenuation=.4)

# Logical to show or hide the viral load in the dynamical plots (will increase computation time in the plotting)
showTrails = True
##########################################


# Method to help with plotting the simulation frames
# Population is a list of agent objects
# Landscape is an environment object
# statsStorage is a list of tuples, where each element is the triple (#healty, #infected, #dead)
# ax is the axis handle
# showTrails is a logical to plot the viral environment, will increase computation / draw time
def populationPlotter(population, landscape, statsStorage, ax, showTrails=False):

    if showTrails:

        # Get coordinates where the viral load is larger than .1, for plotting
        trailsX, trailsY = np.where(landscape.viralMap > .1)
        downScale = 10 ** (-population[0].locationGranularity)
        tx = trailsX*downScale
        ty = trailsY*downScale

        # Set the alpha of the viral trail based on viral load
        alphas = []
        for i, j in zip(trailsX, trailsY):

            alphas.append(landscape.viralMap[i, j])

        # Set the color array: for red the first column needs to be one, alphas are the 4th column
        rgba_colors = np.zeros((len(tx), 4))
        rgba_colors[:, 0] = 1.0
        rgba_colors[:, 3] = alphas

        # Scatter the viral trails
        ax[1].scatter(tx, ty, color=rgba_colors, marker='*')

    # Create arrays for the population stats
    statsX = range(len(statsStorage))
    numInfected = np.array([ss[1] for ss in statsStorage])/populationSize
    numDead = np.array([ss[2] for ss in statsStorage])/populationSize

    # Create stacked color plot based on population proportions
    ax[0].fill_between(statsX, numDead, facecolor='k', alpha=.5, label='Dead')
    ax[0].fill_between(statsX, numDead, numInfected+numDead, facecolor='r', alpha=1, label='Infected')
    ax[0].fill_between(statsX, numInfected+numDead, 1, facecolor='g', alpha=1, label='Healthy')
    ax[0].legend(loc='upper right')

    # Separate colors and locations for the agents
    x = [p.location[0] for p in population]
    y = [p.location[1] for p in population]
    c = [('k' if not p.alive else ('r' if p.infected else 'g')) for p in population]

    # Scatter the agents
    ax[1].scatter(x, y, c=c)

    # Fix the axis limits and title it
    ax[1].set_xlim((0, 1))
    ax[1].set_ylim(0, 1)
    ax[0].set_title('Infection Statistics')


# Seed the infection randomly in the population
patient0 = np.random.randint(0, populationSize)
population[patient0].infected = True
population[patient0].cleanlinessBias = 0
population[patient0].location = np.array([.5, .5])
numInfected = 1
numDead = 0

# Initialize the population outcome statistics list of tuples
statsStorage = [(populationSize-numInfected-numDead, numInfected, numDead)]

# Update environment with the population
landscape.update(population)

# Establish the dynamical plots and perform the first plot
plt.ion()
fig, ax = plt.subplots(2, 1)
populationPlotter(population, landscape, statsStorage, ax, showTrails=showTrails)
plt.show()
plt.pause(.05)
ax[0].clear()
ax[1].clear()

# Extra steps allows the simulation to run for n extra days past the elimination of the infection
extraSteps = 10
maxInfected = 0
while numInfected or extraSteps:

    # Update the population agent by agent
    for a in range(populationSize):

        population[a].update(landscape, population)

    # Update the landscape based on the new population
    landscape.update(population)

    # Stats update
    numInfected = sum([p.alive and p.infected for p in population])
    if numInfected > maxInfected:

        maxInfected = numInfected
    numDead = sum([not p.alive for p in population])
    statsStorage.append((populationSize-numInfected-numDead, numInfected, numDead))

    # Plot
    populationPlotter(population, landscape, statsStorage, ax, showTrails=showTrails)
    plt.show()
    plt.pause(.05)

    # Lagged exit condition
    if numInfected == 0:

        extraSteps -= 1

        if not extraSteps:

            break

    # Clear plots (not executed on the last step in order to keep the final plot around for viewing)
    ax[0].clear()
    ax[1].clear()

plt.ioff()

# Static plots
fig2, ax2 = plt.subplots(2, 1)

# Plot age histogram
ages = [a.age for a in population]
ax2[0].hist([a.age for a in population], density=True)

# Compute percentage of people with pre exiting conditions in each age group, or died in each age group
ageHists, ageBins = np.histogram(ages)
PCvals = np.zeros(len(ageHists), dtype=float)
deathVals = PCvals.copy()
for a in population:

    for bin in range(1, len(ageBins)):

        if a.age < ageBins[bin]:

            if a.preexistingCondition:

                PCvals[bin-1] += 1/float(ageHists[bin-1])

            if not a.alive:

                deathVals[bin-1] += 1/float(ageHists[bin-1])

            break

# Plot the associated graphs
ax2[1].bar(ageBins[:-1], PCvals, width=[(ageBins[i+1] - ageBins[i]) for i in range(len(ageBins)-1)], align='edge')
ax2[0].set_title('Ages')
ax2[1].set_title('Pre-existing Conditions\nLong Illnesses: {0:.0f}%'.format(100*sum([1 if a.infectionTime > 14 else 0 for a in population])/populationSize))
fig2.suptitle('Initial Distributions')

fig3, ax3 = plt.subplots()
ax3.bar(ageBins[:-1], deathVals, width=[(ageBins[i+1] - ageBins[i]) for i in range(len(ageBins)-1)], align='edge')
ax3.set_title('Deaths Given Age: Overall Death rate: {0:.0f}%'.format(100*numDead/maxInfected))
fig3.suptitle('Death Statistics')

plt.show()
