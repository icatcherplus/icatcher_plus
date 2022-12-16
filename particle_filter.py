import numpy as np
import cv2
# code modified from deepgaze, MIT license originally
class ParticleFilter:
    def __init__(self, width, height, N):
        """
        Init the particle filter.
        :param width the width of the frame
        :param height the height of the frame
        :param N the number of particles
        """
        if(N <= 0 or N>(width*height)): 
            raise ValueError('N must be > 0 and <= width*height')
        self.particles = np.empty((N, 2))
        self.particles[:, 0] = np.random.uniform(0, width, size=N)
        self.particles[:, 1] = np.random.uniform(0, height, size=N)
        self.weights = np.array([1.0/N]*N)

    def predict(self, x_velocity, y_velocity, std):
        """
        Predict the position of the point in the next frame.
        :aram x_velocity the velocity of the object along the X axis in terms of pixels/frame
        :param y_velocity the velocity of the object along the Y axis in terms of pixels/frame
        :param std the standard deviation of the gaussian distribution used to add noise
        """
        self.particles[:, 0] += x_velocity + (np.random.randn(len(self.particles)) * std) #predict the X coord
        self.particles[:, 1] += y_velocity + (np.random.randn(len(self.particles)) * std) #predict the Y coord

    def update(self, x, y):
        """
        Update the weights associated which each particle based on the (x,y) coords measured.
        Particles that closely match the measurements give an higher contribution.
        :param x the position of the point in the X axis
        :param y the position of the point in the Y axis
        """
        position = np.empty((len(self.particles), 2))
        position[:, 0].fill(x)
        position[:, 1].fill(y)
        distance = np.linalg.norm(self.particles - position, axis=1)
        max_distance = np.amax(distance)
        distance = np.add(-distance, max_distance)
        self.weights.fill(1.0) #reset the weight array
        self.weights *= distance
        self.weights += 1.e-300 #avoid zeros
        self.weights /= sum(self.weights) #normalize

    def estimate(self):
        """
        Estimate the position of the point given the particle weights.
        :return get the x_mean, y_mean
        """
        x_mean = np.average(self.particles[:, 0], weights=self.weights, axis=0).astype(int)
        y_mean = np.average(self.particles[:, 1], weights=self.weights, axis=0).astype(int)
        return x_mean, y_mean

    def resample(self, method='residual'):
        """
        Resample the particle based on their weights.
        :param method the algorithm to use for the resampling.
            'multinomal' large weights are more likely to be selected [complexity O(n*log(n))]
            'residual' (default value) it ensures that the sampling is uniform across particles [complexity O(N)]
            'stratified' it divides the cumulative sum into N equal subsets, and then 
                selects one particle randomly from each subset.
            'systematic' it divides the cumsum into N subsets, then add a random offset to all the susets
        """
        N = len(self.particles)
        if(method == 'multinomal'):
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1. #avoid round-off error
            indices = np.searchsorted(cumulative_sum, np.random.uniform(low=0.0, high=1.0, size=N))      
        elif(method == 'residual'):
            indices = np.zeros(N, dtype=np.int32)
            # take int(N*w) copies of each weight
            num_copies = (N*np.asarray(self.weights)).astype(int)
            k = 0
            for i in range(N):
                for _ in range(num_copies[i]): # make n copies
                    indices[k] = i
                    k += 1
            #multinormial resample
            residual = self.weights - num_copies     # get fractional part
            residual /= sum(residual)     # normalize
            cumulative_sum = np.cumsum(residual)
            cumulative_sum[-1] = 1. # ensures sum is exactly one
            indices[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))
        elif(method == 'stratified'):
            #N subsets, chose a random position within each one
            #and generate a vector containing this positions
            positions = (np.random.random(N) + range(N)) / N
            #generate the empty indices vector
            indices = np.zeros(N, dtype=np.int32)
            #get the cumulative sum
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1
        elif(method == 'systematic'):
            # make N subsets, choose positions with a random offset
            positions = (np.arange(N) + np.random.random()) / N
            indices = np.zeros(N, dtype=np.int32)
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1
        else:
            raise NotImplementedError("Resampling method not implemented")
        #Create a new set of particles by randomly choosing particles 
        #from the current set according to their weights.
        self.particles[:] = self.particles[indices] #resample according to indices
        self.weights[:] = self.weights[indices]
        #Normalize the new set of particles
        self.weights /= np.sum(self.weights)        

    def returnParticlesContribution(self):
        """
        This function gives an estimation of the number of particles which are
        contributing to the probability distribution (also called the effective N). 
        :return get the effective N value. 
        """
        return 1.0 / np.sum(np.square(self.weights))

    def returnParticlesCoordinates(self, index=-1):
        """
        returns the (x,y) coord of a specific particle or of all particles. 
        :param index the position in the particle array to return
            when negative it returns the whole particles array
        :return a single coordinate (x,y) or the entire array
        """
        if(index<0):
            return self.particles.astype(int)
        else:
            return self.particles[index,:].astype(int)

    def drawParticles(self, frame, color=[0,0,255], radius=2):
        """
        Draw the particles on a frame and return it.
        :param frame the image to draw
        :param color the color in BGR format, ex: [0,0,255] (red)
        :param radius is the radius of the particles
        :return the frame with particles
        """
        for x_particle, y_particle in self.particles.astype(int):
            cv2.circle(frame, (x_particle, y_particle), radius, color, -1)
