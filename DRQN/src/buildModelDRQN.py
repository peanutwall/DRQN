import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
env = gym.make('Frostbite-v0')


class BuildModel:
    def __init__(self, imageLength, imageWidth, numStateSpace, numActionSpace, seed=1):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.imageLength = imageLength
        self.imageWidth = imageWidth
        self.seed = seed

    def __call__(self, sigmoidLayersWidths, tanhLayerWidths, summaryPath="./tbdata"):
        print("Generating LSTM Model with layers: {}, {}".format(sigmoidLayersWidths, tanhLayerWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.imageLength, self.imageWidth, 3], name="states")/ 255
                act_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="act")
                formerOutput_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="formerOutput")
                formerCell_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="formerCell")
                yi_ = tf.placeholder(tf.float32, [None, 1], name="yi")
                tf.add_to_collection("states", states_)
                tf.add_to_collection("act", act_)
                tf.add_to_collection("yi", yi_)
                tf.add_to_collection("formerCell", formerCell_)
                tf.add_to_collection("formerOutput", formerOutput_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.01)

            with tf.variable_scope("convolutionLayers"):
                conv1 = tf.layers.conv2d(
                    inputs=states_,
                    filters=32,
                    kernel_size=[8, 8],
                    strides=4,
                    padding='same',
                    activation=tf.nn.relu
                )
                pool1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=[4, 4],
                    strides=2
                )
                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=64,
                    kernel_size=[4, 4],
                    strides=2,
                    padding='same',
                    activation=tf.nn.relu
                )
                pool2 = tf.layers.max_pooling2d(
                    inputs=conv2,
                    pool_size=[4, 4],
                    strides=2
                )
                conv3 = tf.layers.conv2d(
                    inputs=pool2,
                    filters=64,
                    kernel_size=[3, 3],
                    strides=1,
                    padding='same',
                    activation=tf.nn.relu
                )
                flat = tf.reshape(conv3, [-1, 16*64])
                dense = tf.layers.dense(
                    inputs=flat,
                    units=1024,
                    activation=tf.nn.relu
                )
                dropout = tf.layers.dropout(
                    inputs=dense,
                    rate=0.5
                )
                logits_ = tf.layers.dense(
                    inputs=dropout,
                    units=self.numStateSpace
                )
                inputStates_ = tf.concat([formerOutput_, logits_], 1)

            with tf.variable_scope("forgetSigmoidGate"):
                with tf.variable_scope("trainForgetHidden"):
                    activation_ = inputStates_
                    for i in range(len(sigmoidLayersWidths)):
                        forgetHiddenLayer = tf.layers.Dense(units=sigmoidLayersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="forgetHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = forgetHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{forgetHiddenLayer.kernel.name}"], forgetHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{forgetHiddenLayer.bias.name}"], forgetHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    forgetHiddenOutput_ = tf.identity(activation_, name="forgetHiddenOutput")
                    forgetOutputLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.sigmoid,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="forgetOutputLayer{}".format(len(sigmoidLayersWidths) + 1),
                                                        trainable=True)
                    forgetOutput_ = forgetOutputLayer(forgetHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{forgetOutputLayer.kernel.name}"], forgetOutputLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{forgetOutputLayer.bias.name}"], forgetOutputLayer.bias)
                    tf.add_to_collections("forgetOutput", forgetOutput_)

            with tf.variable_scope("inputSigmoidGate"):
                with tf.variable_scope("trainInputHidden"):
                    activation_ = inputStates_
                    for i in range(len(sigmoidLayersWidths)):
                        inputHiddenLayer = tf.layers.Dense(units=sigmoidLayersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="inputHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = inputHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{inputHiddenLayer.kernel.name}"], inputHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{inputHiddenLayer.bias.name}"], inputHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    inputHiddenOutput_ = tf.identity(activation_, name="inputHiddenOutput")
                    inputOutputLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.sigmoid,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="forgetOutputLayer{}".format(len(sigmoidLayersWidths) + 1),
                                                        trainable=True)
                    inputOutput_ = inputOutputLayer(inputHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{inputOutputLayer.kernel.name}"], inputOutputLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{inputOutputLayer.bias.name}"], inputOutputLayer.bias)
                    tf.add_to_collections("inputOutput", inputOutput_)

            with tf.variable_scope("tanhGate"):
                with tf.variable_scope("trainHiddenTanh"):
                    activation_ = inputStates_
                    for i in range(len(tanhLayerWidths)):
                        tanhHiddenLayer = tf.layers.Dense(units=tanhLayerWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="hiddenTanh{}".format(i + 1),
                                                  trainable=True)
                        activation_ = tanhHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{tanhHiddenLayer.kernel.name}"], tanhHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{tanhHiddenLayer.bias.name}"], tanhHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    tanhHiddenOutput_ = tf.identity(activation_, name="tanhHiddenOutput")
                    outputTanhLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.tanh,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="outputTanh{}".format(len(tanhLayerWidths) + 1),
                                                        trainable=True)
                    tanhOutput_ = outputTanhLayer(tanhHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{outputTanhLayer.kernel.name}"], outputTanhLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{outputTanhLayer.bias.name}"], outputTanhLayer.bias)
                    tf.add_to_collections("tanhOutput", tanhOutput_)

            with tf.variable_scope("opSigmoidGate"):
                with tf.variable_scope("trainOpHidden"):
                    activation_ = inputStates_
                    for i in range(len(sigmoidLayersWidths)):
                        opHiddenLayer = tf.layers.Dense(units=sigmoidLayersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="opHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = opHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{opHiddenLayer.kernel.name}"], opHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{opHiddenLayer.bias.name}"], opHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    opHiddenOutput_ = tf.identity(activation_, name="opHiddenOutput")
                    opOutputLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.sigmoid,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="opOutputLayer{}".format(len(sigmoidLayersWidths) + 1),
                                                        trainable=True)
                    opOutput_ = opOutputLayer(opHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{opOutputLayer.kernel.name}"], opOutputLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{opOutputLayer.bias.name}"], opOutputLayer.bias)
                    tf.add_to_collections("opOutput", opOutput_)

            with tf.variable_scope("trainingParams"):
                learningRate_ = tf.constant(0.001, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)

            with tf.variable_scope("cell"):
                outputCell_ = forgetOutput_*formerCell_+inputOutput_*tanhOutput_
                tf.add_to_collection("outputCell", outputCell_)

            with tf.variable_scope("output"):
                output_ = tf.tanh(outputCell_)*opOutput_
                tf.add_to_collection("output", output_)

            with tf.variable_scope("QTable"):
                QEval_ = tf.reduce_sum(tf.multiply(output_, act_), reduction_indices=1)
                tf.add_to_collections("QEval", QEval_)
                QEval_ = tf.reshape(QEval_, [-1, 1])
                # loss_ = tf.reduce_mean(tf.square(yi_ - QEval_))
                loss_ = tf.reduce_mean(tf.square(yi_ - QEval_))
                # loss_ = tf.losses.mean_squared_error(labels=yi_, predictions=QEval_)
                tf.add_to_collection("loss", loss_)

            with tf.variable_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(loss_)
                tf.add_to_collection("trainOp", trainOpt_)

                saver = tf.train.Saver(max_to_keep=None)
                tf.add_to_collection("saver", saver)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)
            if summaryPath is not None:
                trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
                testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
                tf.add_to_collection("writers", trainWriter)
                tf.add_to_collection("writers", testWriter)
            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            # self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
            #         for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model


def flat(state):
    state = np.concatenate(state, axis=0)
    state = np.concatenate(state, axis=0)
    return state


class CalculateY:

    def __init__(self, model, updateFrequency):
        self.model = model
        self.step = 0
        self.updateFrequency = updateFrequency

    def __call__(self, nextStatesBatch, rewardBatch, doneBatch, gamma, model, formerOutputBatch, formerCellBatch):
        if self.step % self.updateFrequency == 0:
            graph = model.graph
            self.model = model
        else:
            graph = self.model.graph
        self.step += 1
        # graph = model.graph
        output_ = graph.get_collection_ref('output')[0]
        states_ = graph.get_collection_ref("states")[0]
        formerCell_ = graph.get_collection_ref("formerCell")[0]
        formerOutput_ = graph.get_collection_ref("formerOutput")[0]
        outputBatch = self.model.run(output_, feed_dict={states_: nextStatesBatch, formerCell_: formerCellBatch,
                                                    formerOutput_: formerOutputBatch})
        yBatch = []
        for i in range(0, len(nextStatesBatch)):
            done = doneBatch[i][0]
            if done:
                yBatch.append(rewardBatch[i])
            else:
                yBatch.append(rewardBatch[i] + gamma * np.max(outputBatch[i]))
        yBatch = np.asarray(yBatch).reshape(len(nextStatesBatch), -1)
        # print("reward:{}".format(rewardBatch))
        # print("eval:{}".format(evalNetOutputBatch))
        # print("yBatch:{}".format(yBatch))
        return yBatch


class TrainOneStep:

    def __init__(self, batchSize, updateFrequency, learningRate, gamma, calculateY, actionDim):
        self.batchSize = batchSize
        self.updateFrequency = updateFrequency
        self.learningRate = learningRate
        self.gamma = gamma
        self.step = 0
        self.calculateY = calculateY
        self.formerCellBatch = np.random.rand(batchSize, actionDim)
        self.formerOutputBatch = np.random.rand(batchSize, actionDim)

    def __call__(self, model, miniBatch, batchSize):

        # print("ENTER TRAIN")
        graph = model.graph
        yi_ = graph.get_collection_ref("yi")[0]
        act_ = graph.get_collection_ref("act")[0]
        learningRate_ = graph.get_collection_ref("learningRate")[0]
        loss_ = graph.get_collection_ref("loss")[0]
        trainOp_ = graph.get_collection_ref("trainOp")[0]
        output_ = graph.get_collection_ref('output')[0]
        outputCell_ = graph.get_collection_ref("outputCell")[0]
        states_ = graph.get_collection_ref("states")[0]
        formerCell_ = graph.get_collection_ref("formerCell")[0]
        formerOutput_ = graph.get_collection_ref("formerOutput")[0]
        fetches = [loss_, trainOp_]

        states, actions, nextStates, rewards, done = miniBatch
        statesBatch = np.asarray(states).reshape(batchSize, 160, 160, 3)
        actBatch = np.asarray(actions).reshape(batchSize, -1)
        # print("actBatch:{}".format(actBatch))
        nextStatesBatch = np.asarray(nextStates).reshape(batchSize, 160, 160, 3)
        rewardBatch = np.asarray(rewards).reshape(batchSize, -1)
        doneBatch = np.asarray(done).reshape(batchSize, -1)

        outputBatch = model.run(output_, feed_dict={states_: statesBatch, formerCell_: self.formerCellBatch,
                                                         formerOutput_: self.formerOutputBatch})
        cellBatch = model.run(outputCell_, feed_dict={states_: statesBatch, formerCell_: self.formerCellBatch,
                                                           formerOutput_: self.formerOutputBatch})

        yBatch = self.calculateY(nextStatesBatch, rewardBatch, doneBatch, self.gamma, model, self.formerOutputBatch, self.formerCellBatch)
        feedDict = {states_: statesBatch, act_: actBatch, learningRate_: self.learningRate, yi_: yBatch, formerCell_: self.formerCellBatch,
                    formerOutput_: self.formerOutputBatch}
        lossDict, trainOp = model.run(fetches, feed_dict=feedDict)
        self.formerOutputBatch = outputBatch
        self.formerCellBatch = cellBatch

        return model, lossDict


class SampleAction:

    def __init__(self, actionDim):
        self.actionDim = actionDim
        self.formerCell = np.random.rand(1,actionDim)
        self.formerOutput = np.random.rand(1,actionDim)

    def __call__(self, model, states, epsilon):
        if random.random() < epsilon:
            graph = model.graph
            output_ = graph.get_collection_ref('output')[0]
            states_ = graph.get_collection_ref("states")[0]
            outputCell_ = graph.get_collection_ref("outputCell")[0]
            formerCell_ = graph.get_collection_ref("formerCell")[0]
            formerOutput_ = graph.get_collection_ref("formerOutput")[0]
            # states = flat(states)
            output = model.run(output_, feed_dict={states_: [states], formerOutput_: self.formerOutput,
                                                   formerCell_: self.formerCell})
            outputCell = model.run(outputCell_, feed_dict={states_: [states], formerOutput_: self.formerOutput,
                                                   formerCell_: self.formerCell})
            self.formerCell = outputCell
            self.formerOutput = output
            # print("evalNetOutput:{}".format(evalNetOutput))
            # print(np.argmax(QEval[0]))
            # print(evalNetOutput)
            return np.argmax(output)
        else:
            return np.random.randint(0, self.actionDim)


def memorize(replayBuffer, states, act, nextStates, reward, done, actionDim):
    onehotAction = np.zeros(actionDim)
    onehotAction[act] = 1
    replayBuffer.append((states, onehotAction, nextStates, reward, done))
    return replayBuffer


class InitializeReplayBuffer:

    def __init__(self, reset, forwardOneStep, actionDim):
        self.reset = reset
        self.forwardOneStep = forwardOneStep
        self.actionDim = actionDim

    def __call__(self, replayBuffer, maxReplaySize):
        for i in range(maxReplaySize):
            # states = self.reset()
            states = env.reset()
            env.render
            action = np.random.randint(0, self.actionDim)
            # nextStates, reward, done, info = self.forwardOneStep(action)
            nextStates, reward, done, info = env.step(action)
            nextStates = nextStates[34:194]
            states = states[34:194]
            replayBuffer = memorize(replayBuffer, states, action, nextStates, reward, done, self.actionDim)
        return replayBuffer


def sampleData(data, batchSize):
    batch = [list(varBatch) for varBatch in zip(*random.sample(data, batchSize))]
    return batch


def upgradeEpsilon(epsilon):
    epsilon = epsilon + 0.0001*(1-0.5)
    return epsilon


class RunTimeStep:

    def __init__(self, forwardOneStep, sampleAction, trainOneStep, batchSize, epsilon, actionDelay, actionDim):
        self.forwardOneStep = forwardOneStep
        self.sampleAction = sampleAction
        self.trainOneStep = trainOneStep
        self.batchSize = batchSize
        self.actionDelay = actionDelay
        self.epsilon = epsilon
        self.actionDim = actionDim

    def __call__(self, states, model, replayBuffer, score):
        env.render()
        action = self.sampleAction(model, states, self.epsilon)
        print(action)
        for i in range(self.actionDelay):
            self.epsilon = upgradeEpsilon(self.epsilon)
            # nextStates, reward, done, info = self.forwardOneStep(action)
            nextStates, reward, done, info = env.step(action)
            nextStates = nextStates[34:194]

            replayBuffer = memorize(replayBuffer, states, action, nextStates, reward, done, self.actionDim)
            miniBatch = sampleData(replayBuffer, self.batchSize)
            model, loss = self.trainOneStep(model, miniBatch, self.batchSize)
            # print("loss:{}".format(loss))
            print(score)
            score += reward
            states = nextStates
        return states, done, score, replayBuffer, model


class RunEpisode:

    def __init__(self, reset, runTimeStep):
        self.runTimeStep = runTimeStep
        self.reset = reset

    def __call__(self, model, scoreList, replayBuffer, episode):
        # states = self.reset()
        states = env.reset()
        states = states[34:194]
        score = 0
        trajectory = []
        for _ in range(100):
            states, done, score, replayBuffer, model = self.runTimeStep(states, model, replayBuffer, score)
            if done or score > 125:
                scoreList.append(score)
                print('episode:', episode, 'score:', score, 'max:', max(scoreList))
                break
        return model, scoreList, replayBuffer


class RunAlgorithm:

    def __init__(self, episodeRange, runEpisode):
        self.episodeRange = episodeRange
        self.runEpisode = runEpisode

    def __call__(self, model, replayBuffer):
        scoreList = []
        for i in range(self.episodeRange):
            model, scoreList, replayBuffer = self.runEpisode(model, scoreList, replayBuffer, i)
        return model, scoreList

