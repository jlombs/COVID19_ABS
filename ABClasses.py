# Class file for agent based elements

import numpy as np
from scipy.stats import gaussian_kde, gamma
from itertools import product


# This function gives a random age between [0,104] (the age max is 104 throughout the sim).
# A mean <= 0 will use a the 2020 world age distribution, and a positive mean will use a normal distribution centered on
# the mean (still capped within [0,104] using rejection sampling
def age_distribution(mean):

    if mean <= 0:

        # https://www.populationpyramid.net/world/2020/, uses the male distribution (both sexes are very similar and
        # the simulation doesn't include sex variation at this time)
        worldPopulationStats = [349432556, 342927576, 331497486, 316642222, 308286775, 306059387, 309236984, 276447037,
                                249389688, 241232876, 222609691, 192215395, 157180267, 128939392, 87185982, 54754941,
                                33648953, 15756942, 5327866, 1077791, 124144]
        worldPopulationStats = np.array(worldPopulationStats, dtype=float)
        worldPopulationStats /= sum(worldPopulationStats)

        ageGroup = np.random.choice(range(len(worldPopulationStats)), 1, p=worldPopulationStats)[0]
        age = 5*ageGroup + 4*np.random.rand()

    else:

        age = np.random.normal(mean, 10)

        while age < 0 or age > 104:

            age = np.random.normal(mean, 10)

    return age


# This class generates a single agent in the simulation
# ageBias is the mean input to the age_distribution function
# cleanlinessBias [0,1] gives how effective the agent is at cleaning their environment of the virus
# socialDistanceBias [0,1] gives how effective the agent is at making good social distancing choices when they move
# travelerBias [0, 10] gives how energetic the agent is when they travel about. 1 will give appropriately scaled motion,
#     >1 indicates 'extra active'
# locationGranularity is used to set the discrete simulation scale, where 10**granularity is the 1D volume. 2 is ideal
# initialLocationBias can be used to bias the initial location of the agent within the environment. Uniformly random
#     choices will be used if none is provided. Else, provide a 2D array within the [0,1]^2 square
class agent:

    def __init__(self, ageBias=-1, cleanlinessBias=0, socialDistanceBias=0, travelerBias=1, locationGranularity=2,
                 initialLocationBias=None):

        # Save initializations in case they will be used
        self.ageBias = ageBias
        self.cleanlinessBias = cleanlinessBias
        self.socialDistanceBias = socialDistanceBias
        self.travelerBias = travelerBias
        self.travelerBiasO = travelerBias
        self.locationGranularity = locationGranularity
        self.initialLocationBias = initialLocationBias

        # Assign agent age
        self.age = age_distribution(self.ageBias)

        # Assign initial location
        if self.initialLocationBias:

            location = np.random.normal(initialLocationBias, 10**(-self.locationGranularity)).round(locationGranularity)
            location = np.array([min(max(dim, 0), 1) for dim in location])

        else:

            location = np.random.random(2).round(locationGranularity)

        self.location = location.copy()

        # Assign if the agent has a preexisting condition. We use the gamma cdf where a 20 yr old has a
        # .2% chance, and a 100 year old has a 99% chance. A 50 yr old has a 43% chance.
        self.preexistingCondition = np.random.rand() < gamma(12, scale=4.5).cdf(self.age)

        # Set initial state variables: currently infected, symptomatic, alive, or protected from future infection
        self.infected = False
        self.symptomatic = False
        self.alive = True
        self.protected = False

        # Set infection length, in days, if the agent were to get sick
        # If preexisting condition, 4 weeks of sickness. Else, 2 weeks unless elderly, then 4 again regardless of PEC
        # Elderly is determined based on the same gamma cdf as preexisting conditions
        if self.preexistingCondition or np.random.rand() < gamma(12, scale=4.5).cdf(self.age):

            self.infectionTime = 7*4

        else:

            self.infectionTime = 7*2

        self.infectionTimer = self.infectionTime

        # Set the time the agent would be asymptomatic for. Uses a gamma distribution with median 5 days and tails out
        # to ~28 days, as per https://www.worldometers.info/coronavirus/
        self.asymptomaticTime = np.random.gamma(3, 2)
        self.asymptomaticTimer = self.asymptomaticTime

    # Method for updating the agent: takes in an environment object and a list of agent objects as a population
    def update(self, landscape, population):

        # If alive, move to a new location
        if self.alive:

            # If social distancing not active, gather 1 candidate location, else gather 10
            testLocations = []

            if np.random.rand() > self.socialDistanceBias:

                tests = 1

            else:

                # Construct a gaussian KDE based on the 10 nearest neighbors in Euclidean 2-norm to the current agent
                # Note, this is expensive, so if the population is >500 agents and social distancing propensity is
                # strong, this will take time. Look for ways to speed this up!
                tests = 10
                locations = np.array([p.location for p in population])
                closestFriends = locations[np.argsort([sum((l-self.location)**2) for l in locations])[1:11]]
                kde = gaussian_kde(closestFriends.T)

            # Generate new locations based on random normal perturbations at the travelerBias influenced scale centered
            # on the current location, using rejection sampling for boundary handling
            while len(testLocations) < tests:

                newLocation = np.random.normal(self.location, self.travelerBias*10**(-self.locationGranularity)).\
                    round(self.locationGranularity)

                while not (np.all(0 < newLocation) and np.all(newLocation < 1)):

                    newLocation = np.random.normal(self.location, self.travelerBias*10**(-self.locationGranularity)).\
                        round(self.locationGranularity)

                testLocations.append(newLocation.copy())

            # Take the lowest scoring point based on the PDF of the KDE, or the only point in the non-SD case
            if tests > 1:

                bestLocation = np.argmin(kde.evaluate(np.array(testLocations).T))

            else:

                bestLocation = 0

            self.location = testLocations[bestLocation].copy()

            # If no immunity and not currently infected, get a chance to become infected
            # Note, immunity loss or reinfection is not coded atm, but will come soon as new COVID data is understood
            if not self.infected and not self.protected:

                # Environmental infection
                # Get coordinates of current location
                i, j = int(self.location[0]*landscape.scale), int(self.location[1]*landscape.scale)
                # If environmental viral load is high, more likely for infection
                if np.random.rand() < landscape.viralMap[i, j]:

                    self.infected = True

            # If infected
            elif self.infected:

                # Count down until symptomatic
                if self.asymptomaticTimer > 0:

                    self.asymptomaticTimer -= 1

                # Symptomatic
                else:

                    # If symptomatic, reduce movement during recovery
                    self.travelerBias = .5

                    # Count down until infection is healed
                    self.infectionTimer -= 1

                    # Healed
                    if self.infectionTimer == 0:

                        self.infected = False
                        self.protected = True  # Currently everyone gets immunity after infection

                        # Reset timers, to be used in the future
                        self.infectionTimer = self.infectionTime
                        self.asymptomaticTimer = self.asymptomaticTime
                        self.travelerBias = self.travelerBiasO

                    # If not healed yet
                    else:

                        # Death chance, comorbid, based on https://www.worldometers.info/coronavirus/
                        # We use the 10.5% maximum death likelihood for comorbidities, spread over the infection period
                        # Each day there is some uniform probability of death based on the known aggregates

                        if self.preexistingCondition:

                            deathFactor = .105

                        else:

                            deathFactor = 0

                        # If no underlying issues, we default to using age statistics
                        # elif not self.preexistingCondition:

                        if self.age <= 39:

                            ageFactor = .002

                        elif 40 <= self.age <= 49:

                            ageFactor = .004

                        elif 50 <= self.age <= 59:

                            ageFactor = .013

                        elif 60 <= self.age <= 69:

                            ageFactor = .036

                        elif 70 <= self.age <= 79:

                            ageFactor = .08

                        else:

                            ageFactor = .148

                        if np.random.rand() >= (1-max(ageFactor, deathFactor))**(1/self.infectionTime):

                            self.alive = False


# An environment class to house the infection landscape
# locationGranularity is used to set the discrete simulation scale, where 10**granularity is the 1D volume
# attenuation in [0, 1] gives the multiplicative rate against which the virus decays in the wild. Each day, undisturbed
#     environmental virus rates are multiplied by attenuation. Maximum load is 1; therefore after 3 days, the load is
#     .4**3 = .06 under the defaults.
#     https://www.cnbc.com/2020/03/18/coronavirus-lives-for-hours-in-air-particles-and-days-on-surfaces-new-us-study-shows.html
# AOE in [1, n] is the area of effect default for how far one agent spreads the virus around them.
#     AOE**2 is the granular area
class environment:

    def __init__(self, locationGranularity=2, attenuation=.4, AOE=10):

        # Save initializations in case they will be used
        self.locationGranularity = locationGranularity
        self.attenuation = attenuation
        self.AOE = AOE

        # Set the environment scale
        self.scale = 10 ** self.locationGranularity

        # Establish the viral load map as a matrix
        self.viralMap = np.zeros((self.scale+1, self.scale+1))

    # Method to update the environment based on the current population, a list of agent objects
    def update(self, population):

        # Iterate over agents
        for p in population:

            # If the agent is alive and either infected or cleans, update the viral load map
            if p.alive and (p.infected or p.cleanlinessBias):

                # Get the agent location and coordinates
                loc = p.location
                i, j = int(loc[0]*self.scale), int(loc[1]*self.scale)

                # Examine the AOE block surrounding the agent
                for ii, jj in product(range(i-self.AOE, i+self.AOE+1), range(j-self.AOE, j+self.AOE+1)):

                    # Ignore points off the grid
                    if 0 <= ii <= self.scale and 0 <= jj <= self.scale:

                        # Compute the taxicab distance
                        d = abs(ii-i) + abs(jj-j)

                        # Increase the viral load (up to a maximum of 1) based on distance from agent in the AOE
                        # Load decreases with the same attenuation factor exponentially with distance
                        # The agent can also decrease viral load in the same fashion according to their cleanlinessBias
                        self.viralMap[ii, jj] = min(self.viralMap[ii, jj]+p.infected*self.attenuation**(d/2), 1)
                        self.viralMap[ii, jj] = max(self.viralMap[ii, jj]-p.cleanlinessBias*self.attenuation**(d/2), 0)

        # Attenuate the viral map
        self.viralMap *= self.attenuation
