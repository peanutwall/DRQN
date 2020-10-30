import tensorflow as tf
import numpy as np
import sys
from collections import deque
import os
import gym
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
from src.buildModelDRQN import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imageLength = 160
imageWidth = 160
stateDim = 10
actionDim = 6
seed = 128
bufferSize = 100000
maxReplaySize = 256
batchSize = 128
gamma = 0.95
epsilon = 0.5
learningRate = 0.000001
sigmoidLayerWidths = [30, 30]
tanhLayerWidths = [30, 30]
scoreList = []
episodeRange = 100
actionDelay = 1
updateFrequency = 6

env = gym.make('Pong-v0')

reset = env.reset
forwardOneStep = env.step
sampleAction = SampleAction(actionDim)
initializeReplayBuffer = InitializeReplayBuffer(reset, forwardOneStep, actionDim)
buildModel = BuildModel(imageLength, imageWidth, stateDim, actionDim)

model = buildModel(sigmoidLayerWidths, tanhLayerWidths)

calculateY = CalculateY(model, updateFrequency)
trainOneStep = TrainOneStep(batchSize, updateFrequency, learningRate, gamma, calculateY, actionDim)
replayBuffer = deque(maxlen=bufferSize)
replayBuffer = initializeReplayBuffer(replayBuffer, maxReplaySize)
miniBatch = sampleData(replayBuffer, batchSize)

runTimeStep = RunTimeStep(forwardOneStep, sampleAction, trainOneStep, batchSize, epsilon, actionDelay, actionDim)
runEpisode = RunEpisode(reset, runTimeStep)
runAlgorithm = RunAlgorithm(episodeRange, runEpisode)
model, scoreList = runAlgorithm(model, replayBuffer)