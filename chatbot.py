"""
############################################################################################
Main Classes:
	Environment				The environment for the agent to explore
	ReplayMemory				The memory which holds the states, action taken and Q-values	
	NeuralNetwork				The neural network for estimating Q-values
	Agent					The agent to learn the environment		
############################################################################################
Pseudocode:
	1. The neural network is created, and the Q-values are initialised

	2. The environment creates an array of questions (a conversation)

	3. Get the word vector for the next sentence in the conversation array (and previous state).
	The state in this a word vector built using Word2Vec and a previous state of last answer

	4. Using the epsilon greedy at that given point in time, select an action. This either
	takes a random action in a reduced action space or takes the max value (based on the 
	episilon greedy)

	5. The reward and end of episode boolean are returned. These are used to calculate the
	Q-values and update the replay memory. These memories are then stored

	6. If the replay memory is completely full, the replay memory is updated in reverse

	7. If step 6 completes. The neural network is then optimised using a random batch 
	from the replay memory

	8. Move to the next question in the conversation, when the end of episode boolean
	is true, get a new conversation

	9. A checkpoint is saved for use in testing when the number of episodes is reached
############################################################################################
Imports:
	OS: 					Changing dir
	Tensorflow: 				Creating the neural network
	W2V: 					Vector representations of the words
	Gensim: 				Modelling which contains W2V with additional functionality
	NLTK - word_tokenize			Allows to build a sentence vector
	Argparse:				For passing arguments into cmd
	Time:					Used during testing when the user enters a goodbye message
	Sys:					Clearing the screen for the training screen
	timeit - default_timer			Used for timing the execution time of training
	matplotlib.pyplot			Plotting average reward over time
############################################################################################
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import gensim
from nltk import word_tokenize
import argparse
import time
import sys
from timeit import default_timer as timer
import matplotlib.pyplot as plt
############################################################################################

# Set up the checkpoint and log directories
checkpoint_directory = 'C:/Users/Keelan/Desktop/Project/Checkpoints'
log_directory = 'C:/Users/Keelan/Desktop/Project/Logs'

# Load pre-trained Word2Vec model created by the create_model class
os.chdir("C:/Users/Keelan/Desktop/Project/Vec_Models")
model = gensim.models.Word2Vec.load('conversations')

# Load the conversation headers from the CSV file
os.chdir("C:/Users/Keelan/Desktop/Project/Conversations")
textdata = pd.read_csv('Conversations.csv')
greetings = textdata['Greetings'].values.tolist()
greetings_answer = textdata['Greetings_A'].values.tolist()
place = textdata['Place'].values.tolist()
place_answer = textdata['Place_A'].values.tolist()
cost = textdata['Cost'].values.tolist()
answers = textdata['Answers'].values.tolist()

# Remove NaN values in columns
greetings = [x for x in greetings if str(x) != 'nan']
greetings_answer = [x for x in greetings_answer if str(x) != 'nan']
place = [x for x in place if str(x) != 'nan']
place_answer = [x for x in place_answer if str(x) != 'nan']
cost = [x for x in cost if str(x) != 'nan']



class Environment:
	"""
	Description: 
		This creates the conversation and acts as the user for training by being able
		to return the state and it defines a step within the environment.
		It returns the reward and end_of_episode boolean to the Agent.
	Parameters:
		N/A
	"""

	def create_conversation(self):
		# This creates the conversation for the user simulator

		# Get the greeting
		hello = greetings[np.random.randint(0,len(greetings))]

		# Select a cost and a place
		place_n = np.random.randint(0,(len(place)))
		cost_n = np.random.randint(0,(len(cost)))

		# Select the relevant corresponding answer based on the place and the cost
		# e.g. "Italian" and "Expensive" selects an "Expensive Italian Restaurant"
		# as the answer 
		answer_n = ((len(cost)*place_n) + (cost_n +1)) - 1

		# Set the answer in the conversation
		self.answer = answers[answer_n]

		# Create conversation array
		self.conversation = [hello, place[place_n], cost[cost_n], self.answer]

		# Count the correct answers in the conversation - for issuing a reward
		self.correct_answers = 0

		return self.conversation

	def get_state(self, curr_question, prev_question = ""):
		"""
		Description:
			State to input to Neural Network.
			This is a numerical representation of the question.
			It consists of the word vector of the current input and the previous input.

			Note: Word2Vec is the first 2 layers of the Neural network.
			This is because word2vec is 2 layer NN that processes text

			Inpsired by https://github.com/stanleyfok/sentence2vec
		Parameters:
			curr_question: the current question in sentence format
		Return:
			The state (float vector of size 1)
		"""

		# For each modelled word in the sentence, sum the vectors
		# NOTE: NOT USED as vector size = 1
		vectors = [model.wv[w] for w in word_tokenize(curr_question.lower())
					if w in model.wv]
		state = np.zeros(model.vector_size)
		# Get the average of the vector sum
		state = (np.array([sum(x) for x in zip(*vectors)])) / state.size
		
		# If there is a previous question
		if(prev_question != ""):
			# Same as above
			vectors2 = [model.wv[w] for w in word_tokenize(prev_question.lower())
					if w in model.wv]
			last_state = np.zeros(model.vector_size)
			last_state = (np.array([sum(x) for x in zip(*vectors2)])) / last_state.size

		# If not, just set the vector to 0 (first question in conversation)
		else:
			last_state = np.zeros([1])

		# Add the current state and the previous together to form one state
		this_state = np.concatenate([state, last_state])

		return this_state

	def total_actions(self):
		# Add the action columns
		num_actions = (len(greetings_answer) + len(place_answer) + len(answers))

		return num_actions

	def action_space_size(self, curr_question):
		# Get the number of actions (replies)

		# Add the answer columns in conversations.csv
		num_actions = 0
		if(curr_question == 0):
			num_actions = len(greetings_answer)
		elif(curr_question == 1):
			num_actions = len(place_answer)
		else:
			num_actions = len(answers)

		return num_actions

	def step(self, curr_question, action):
		"""
		Description:
			This is when the agent takes a step in the environment
		Parameters:
			curr_question:	The current question(or step) in the conversation array
			action:			The action selected by the Agent
			attempt_number:	Current attempt number at answering this question (increment if wrong)	
		Return:
			Returns the reward, whether it was the end of the conversation, and whether the answer was correct
		"""

		# The possible actions of the chatbot
		possible_actions = greetings + place + answers

		"""
		If the current answer is correct
		Note: This will not be correct for curr_question == action_taken == 2
		This is because in the conversations array, [2] is the cost
		"""
		if(self.conversation[curr_question] == possible_actions[action]):
			self.correct_answers+=1
			reward = 0.2
			end_episode = False

		# When the action taken == 2 (Cost)
		elif(curr_question == 2):\
			# Correct answer for the ENTIRE conversation
			if(self.conversation[3] == possible_actions[action]):
				reward = 0.2
				# Check if the rest of the conversation was also correct
				if(self.correct_answers == 2):
					# Fully correct sequence. +1 reward total
					reward = 0.6
				end_episode = True
			else:
				# Incorrect
				reward = 0.0
				end_episode = True
				
		# If the answer was incorrect
		else:
			reward = 0.0
			end_episode = False

		return reward, end_episode



class ReplayMemory:
	"""
	Description:
		This class holds many previous states of the environment

	Parameters:
		size: size of the replay memory (states).
		num_actions: Number of possible actions in the environment.
		discount_rate: The discount factor for updating Q-values.

	# Inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials
	# Updating Q-Values, getting batch indeces and some other parts are vastly different
	"""

	def __init__(self, size, num_actions, discount_rate=0.95):

		# Size of the states (last 2 input sentences)
		self.state_shape = [2]

		# Previous states
		self.states = np.zeros(shape=[size] + self.state_shape, dtype=np.float)

		# Array for the Q-values corresponding to the states
		self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

		# Old Q-value array for comparing
		self.old_q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

		# Holds the actions corresponding to the states
		self.actions = np.zeros(shape=size, dtype=np.int)

		# Holds the rewards corresponding to the states
		self.rewards = np.zeros(shape=size, dtype=np.float)

		# Whether the conversation has ended (end of episode)
		self.end_episode = np.zeros(shape=size, dtype=np.bool)

		# Number of states
		self.size = size

		# Discount factor per step
		self.discount_rate = discount_rate

		# Reset the current size of the replay memory
		self.current_size = 0

	def is_full(self):
		#Used to check if the replay memory is full

		return self.current_size == self.size

	def reset_size(self):
		# Empty the replay memory
		self.current_size = 0

	def add_memory(self, state, q_values, action, reward, end_episode):
		# Add a state to the replay memory. The parameters are all the things we want to store

		# Move to current index and increment size
		curr = self.current_size
		self.current_size += 1

		# Store
		self.states[curr] = state
		self.q_values[curr] = q_values
		self.actions[curr] = action
		self.end_episode[curr] = end_episode
		self.rewards[curr] = reward
		# Clip reward to between -1.0 and 1.0
		#self.rewards[curr] = np.clip(reward, -1.0, 1.0)

	def update_q_values(self):
		# Update the Q-values in the replay memory

		# Keep old q-values
		self.old_q_values[:] = self.q_values[:]

		# Update the Q-values in a backwards loop
		for curr in np.flip(range(self.current_size-1),0):

			# Get data from curr
			action = self.actions[curr]
			reward = self.rewards[curr]
			end_episode = self.end_episode[curr]

			# Calculate Q-Value
			if end_episode:
				# No future steps therefore it is just the observed reward
				value = reward
			else:
				# Discounted future rewards
				value = reward + self.discount_rate * np.max(self.q_values[curr + 1])

			# Update Q-values with better estimate
			self.q_values[curr, action] = value

	def get_batch_indices(self, batch_size):
		# Get random indices from the replay memory (number = batch_size)

		self.indeces = np.random.choice(self.current_size, size=batch_size, replace=False)

	def get_batch_values(self):
		# Get the states and q values for these indeces

		batch_states = self.states[self.indeces]
		batch_q_values = self.q_values[self.indeces]

		return batch_states, batch_q_values



class NeuralNetwork:
	"""
	Description: 
		This implements the neural network for Q-learning.
		The neural network estimates Q-values for a given state so the agent can decide which action to take.
	Parameters:
		num_actions: The number of actions (number of Q-values needed to estimate)
		replay_memory: This is used to optimize the neural network and produce better Q-values

	# Init values and optimization inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials
	"""

	def __init__(self, num_actions, replay_memory):

		# Reset default graph
		tf.reset_default_graph()

		# Path for saving checkpoints during training for testing
		self.checkpoint_dir = os.path.join(checkpoint_directory, "checkpoint")

		# Size of state (2x Word2Vec vectors)
		self.state_shape = [2]

		# Sample random batches
		self.replay_memory = replay_memory

		# Inputting states into the neural network
		with tf.name_scope("inputs"):
			self.states = tf.placeholder(dtype=tf.float32, shape=[None] + self.state_shape, name='state')

		# Learning rate placeholder
		self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

		# Input the new Q-values placeholder that we want the states to map to
		self.q_values_new = tf.placeholder(dtype=tf.float32, shape=[None, num_actions], name='new_q_values')

		# Initialise weights close to 0
		weights = tf.truncated_normal_initializer(mean=0.0, stddev=1e-1)

		# Hidden Layer 1
		# Note: This takes the word2vec state as input
		layer1 = tf.layers.dense(inputs=self.states, name='hidden_layer1', units=20, kernel_initializer=weights, activation=tf.nn.relu)

		# Hidden Layer 2
		layer2 = tf.layers.dense(inputs=layer1, name='hidden_layer2', units=20, kernel_initializer=weights, activation=tf.nn.relu)

		# Output layer - estimated Q-values for each action
		output_layer = tf.layers.dense(inputs=layer2, name='output_layer', units=num_actions, kernel_initializer=weights, activation=None)

		# Set the Q-values equal to the output from the output layer
		with tf.name_scope('Q-values'):
			self.q_values = output_layer
			tf.summary.histogram("Q-values", self.q_values)

		# Get the loss
		# Note: This is the mean-squared error between the old and new Q-values (L2-Regression)
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.q_values - self.q_values_new), axis = 1))
			tf.summary.scalar("loss", self.loss)

		# Optimiser for minimising the loss (to learn better Q-values)
		with tf.name_scope('optimizer'):
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		# Create TF session for running NN
		self.session = tf.Session()

		# Merge all summaries for Tensorboard
		self.merged = tf.summary.merge_all()

		# Create Tensorboard session
		self.writer = tf.summary.FileWriter(log_directory, self.session.graph)

		# Initalialise all variables and run
		init = tf.global_variables_initializer()
		self.session.run(init)

		# For saving the neural network at the end of training (for testing)
		self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

	def close(self):
		# Close TF Session
		self.session.close()

	def get_q_values(self, states):
		# Calculate and return estimated Q-values for the given states
		
		# Get estimated Q-values from the neural network
		q_values = self.session.run(self.q_values, feed_dict={self.states: states})

		return q_values

	def optimize(self, current_state, learning_rate, batch_size=50):
		"""
		Description: 
			This is the optimization function for the neural network. This updates the Q-values
			from a random batch using the learning rate.
		Parameters:
			current_state: The current state being processed when the optimize function is called
			learning_rate: The learning rate of the neural network
			batch_size: The size of the batch taken from the replay memory
		"""
		print("Optimization of Neural Network in progress with learning rate {0}".format(learning_rate))
		
		# Get random indices from the replay memory
		self.replay_memory.get_batch_indices(batch_size)

		# Get the corresponding states and q values for the indices
		batch_states, batch_q_values = self.replay_memory.get_batch_values()

		# Feed these values into the neural network and run one optimization and get the loss value
		current_loss, _ = self.session.run([self.loss, self.optimizer], feed_dict = {self.states: batch_states, self.q_values_new: batch_q_values, self.learning_rate: learning_rate})

		# Send the results to tensorboard
		result = self.session.run(self.merged, feed_dict={self.states: batch_states, self.q_values_new: batch_q_values, self.learning_rate: learning_rate})
		print("Current loss: ", current_loss)
		self.writer.add_summary(result, current_state)

	def save(self, count_states):
		# Save the completed trained network
		self.saver.save(self.session, save_path=self.checkpoint_dir, global_step=count_states)
		print("Checkpoint saved")

	def load(self):
		# Load the network for testing
		try:
			latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_directory)
			self.saver.restore(self.session, save_path=latest_checkpoint)
			print("\n\nLoad of neural network successful. Please begin to talk to the chatbot!:")
		except:
			print("Could not find checkpoint")

class Agent:
	"""
	Description: 
		Agent that replies to the user simulator. Receives S,R -> Action output
	Parameters:
		training: training or testing. This is for run()
	"""

	def __init__(self, training):

		# Create the environment
		self.environment = Environment()

		# Training or testing
		self.training = training

		# Set the initial training epsilon
		self.epsilon = 0.10

		# Get the number of actions for storing memories and Q-values etc.
		total_actions = self.environment.total_actions()

		# Training or testing
		if self.training:
			# Training : Set a learning rate
			self.learning_rate = 1e-2

			# Training: Set up the replay memory
			self.replay_memory = ReplayMemory(size=1000, num_actions=total_actions)

		else:
			# Testing: These are not needed
			self.learning_rate = None
			self.replay_memory = None

		# Create the neural network
		self.neural_network = NeuralNetwork(num_actions=total_actions, replay_memory=self.replay_memory)

		# This stores the rewards for each episode
		self.rewards = []

	def get_action(self, q_values, curr_question):
		"""
		Description:
			Use the current epsilon greedy to select an action
		Parameters:
			q_values: q_values at current state
			iteration_count: count of processed states
			training: training or testing
		Return:
			action: the selected reply
		"""

		# This is used when a random reply is selected. It encourages more efficient interactions when selecting randomly
		# as it only lets the chatbot select from the correct column (this speeds up training a bit by letting the 
		# agent find the correct answers a little more easily).
		if(curr_question) == 0:
			low = 0
			high = len(greetings)
		elif(curr_question) == 1:
			low = len(greetings)
			high = len(greetings) + len(place)
		else:
			low = len(greetings) + len(place)
			high = len(greetings) + len(place) + len(answers)

		# self.epsilon = probability of selecting a random action
		if np.random.random() < self.epsilon:
			# Random sentence reply in the correct column
			action = np.random.randint(low=low, high=high)
		else:
			# Select highest Q-value
			action = np.argmax(q_values)

		return action

	def get_testing_action(self, q_values):

		# During testing, always select the maximum Q-value
		action = np.argmax(q_values)

		return action

	def run(self, num_episodes=1000000):
		"""
		Description:
			Run the agent in either training or testing mode
		Parameters:
			num_episodes: The number of episodes the agent will run for in training mode

		# Loosely inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials
		"""

		if self.training:

			# Reset following loop
			end_episode = True

			# Counter for the states processed so far
			count_states = 0

			# Counter for the episodes processed so far
			count_episodes = 0

			# Counter for which step of the conversation (question)
			conversation_step = 0

			while count_episodes <= num_episodes:
				if end_episode:
					# Generate new conversation for the new episode
					conversation = self.environment.create_conversation()

					# The number of questions for this episode
					num_questions = len(conversation)

					# Reset conversation step
					conversation_step = 0				

					# Increment count_episodes as it is the end of the conversation
					count_episodes += 1

					# Reset episode reward
					reward_episode = 0.0

					if count_episodes > num_episodes:
						self.neural_network.save(count_states)

				if(conversation_step == 0):
					# First step in conversation. No previous question
					state = self.environment.get_state(curr_question = conversation[conversation_step])
				else:
					# Pass in the prev
					prev_question_idx = conversation_step - 1
					prev_question = conversation[prev_question_idx]
					state = self.environment.get_state(curr_question = conversation[conversation_step], prev_question = prev_question)

				# Estimate Q-Values for this state
				q_values = self.neural_network.get_q_values(states=[state])[0]

				# Determine the action
				action = self.get_action(q_values=q_values, curr_question = conversation_step)
				
				# Use action to take a step / reply
				reward, end_episode = self.environment.step(curr_question = conversation_step, action=action)

				# Increment to the next conversation step
				conversation_step += 1

				# Add to the reward for this episode
				reward_episode += reward

				# Increment the episode counter for calculating the control parameters
				count_states += 1

				# Add this memory to the replay memory
				self.replay_memory.add_memory(state=state,q_values=q_values,action=action,reward=reward,end_episode=end_episode)

				if self.replay_memory.is_full():
					# If the replay memory is full, update all the Q-values in a backwards sweep
					self.replay_memory.update_q_values()

					# Improve the policy with random batches from the replay memory
					self.neural_network.optimize(learning_rate=self.learning_rate, current_state=count_states)

					# Reset the replay memory
					self.replay_memory.reset_size()

				# Add the reward of the episode to the rewards array 
				if end_episode:
					self.rewards.append(reward_episode)

				# Reward from previous episodes (mean of last 30)
				if len(self.rewards) == 0:
					# No previous rewards
					reward_mean = 0.0
				else:
					# Get the mean of the last 30
					reward_mean = np.mean(self.rewards[-30:])

				if end_episode:
					# Print statistics
					statistics = "{0:4}:{1}\tReward: {2:.1f}\tMean Reward (last 30): {3:.1f}\tQ-min: {4:5.7f}\tQ-max: {5:5.7f}"
					print(statistics.format(count_episodes, count_states, reward_episode, reward_mean, np.min(q_values), np.max(q_values)))


		# TESTING
		else:
			# Clear cmd window and print chatbot intro
			clear = lambda: os.system('cls')
			clear()
			time.sleep(0.5)
			print()
			print("\t\t        __          __  __          __ ")
			time.sleep(0.5)
			print("\t\t  _____/ /_  ____ _/ /_/ /_  ____  / /_")
			time.sleep(0.5)
			print("\t\t / ___/ __ \/ __ `/ __/ __ \/ __ \/ __/")
			time.sleep(0.5)
			print("\t\t/ /__/ / / / /_/ / /_/ /_/ / /_/ / /_  ")
			time.sleep(0.5)
			print("\t\t\___/_/ /_/\__,_/\__/_.___/\____/\__/  ")
			time.sleep(0.5)

			# Load the conversation checkpoint generated by training
			self.neural_network.load()

			# Set the previous question to blank so it returns a word vector of 0
			previous_question = ""

			# Current question counter
			curr_question = 0

			while True:

				# Get the next user question/input
				user_input = input("User: ").lower()
				try:

					# Get the state for this input
					if(previous_question == ""):
						# First question
						state = self.environment.get_state(curr_question=user_input)
					else:
						state = self.environment.get_state(curr_question=user_input, prev_question=previous_question)

					print("STATE:",state)

					# Input this question into the neural network
					q_values = self.neural_network.get_q_values(states=[state])[0]

					# Store previous question
					previous_question = user_input

					# Possible actions of agent (replies)
					possible_actions = greetings_answer + place_answer + answers

					print("Q VALUES:",q_values)

					# Select an action based on the q-values
					action = self.get_testing_action(q_values = q_values)

					print("Chatbot: ", possible_actions[action])
					if(curr_question < 2):
						curr_question += 1
					else:
						print("=EXPECTED END OF CONVERSATION. CONVERSATION RESTARTED=")
						# Reset
						curr_question = 0
						previous_question = ""

				except:
					print("This sentence is not recognised")

				# Closes the system
				if user_input == "bye":
					print("Closing chatbot in 5...")
					time.sleep(5)
					break


if __name__ == '__main__':
	# Running the system from command line

	# Parsing for command line
	description = "Q-Learning chatbot"

	# Create parser and add arguments
	parser = argparse.ArgumentParser(description=description)

	# Training argument: add "-training" to run training 
	parser.add_argument("-training", required=False,
						dest='training', action='store_true',
						help="train or test agent")

	# Parse the args
	args = parser.parse_args()
	training = args.training

	# Take note of the time taken to train the system
	start = timer()

	# Create and run the agent
	agent = Agent(training=training)
	agent.run()

	# Calculate time taken
	end = timer()
	time_taken = ((end - start)/60)

	# Get the rewards
	rewards = agent.rewards

	# Print statistics about the rewards
	if training:
		print("#############################################################")
		# Print the number of episodes
		print("Statistics for {0} episodes:".format(len(rewards)))
		# Number of instances of maximum reward (i.e. all answers correct)
		print("No. of correct sequences:\t", 	rewards.count(np.max(rewards)))
		# Print first occurrence
		print("First occurrence:\t\t", 			rewards.index(np.max(rewards)))
		# Mean reward for all occurences
		print("Mean Reward:\t\t\t",				np.mean(rewards))
		# Print time taken
		print("Time taken(m):\t\t\t",			time_taken)

		# Plot the total average reward over time
		rewards_plot = []
		for x in range(len(rewards)):
			if (x>0 and (x % 10000 == 0)):
				# This gets the average reward of the last 10000 results
				values = np.mean(rewards[(x-10000):x])
				rewards_plot.append(values)


		# Plot the reward over time
		plt.title("Graph of Total Average Reward Over Time")
		plt.ylabel("Average Reward")
		plt.xlabel("Number of episodes /1000 episodes")
		plt.plot(rewards_plot, 'r')
		plt.show()
