# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* To quickly summarise, this repository is used to store my work on my bachelor thesis for future reference. As this repository was created in the last 2 weeks that I was able to work on the repository, there should not be many commits other than updates to the report and presentations. Furthermore, this repository is not intended to be public, and should stay private. The only intended viewers are myself, and (potentially) academics at Tohoku University and Macquarie University.
* Version 1.0


### Implementation ###
* The code was implemented in Python on a Linux System. 
* The ViZDoom framework and tensorflow were used. 
* While it is (supposedly) possible to implement on a Windows system, it is not recommended.

### Source Acknowledgements ###

# Project Proposal and Learning Resources
The existence of this project is largely due to the opportunity provided to me by the Shinohara Laboratory at Tohoku University.
Yutaro Kikuchi (of the same lab) was very prominant in this project, as he mentored me throughout the 15 weeks in the lab, and was originally working with A3C and ViZDoom before I arrived.

# Algorithms and ViZDoom
The algorithms used in this project are A3C and A2C, developed by Google's DeepMind and OpenAI respectively. The respective papers and sources can be found here
	A3C: 	'Asynchronous Methods for Deep Reinforcement Learning' (2016), https://arxiv.org/pdf/1602.01783.pdf
	A2C:	'Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation' (2017), https://arxiv.org/pdf/1708.05144.pdf
			'OpenAI Baselines: ACKTR & A2C' (2017), https://blog.openai.com/baselines-acktr-a2c/
			
ViZDoom is an AI learning platform that provides a customizable and accesible framework for training an AI to play DOOM. Their website has several videos, tutorials, installation instructions and a link to their GitHub. 
	http://vizdoom.cs.put.edu.pl/

# Implementations
The code used in this project is heavily derived off of Arthur Julianji's implementation (found in his GitHub here: https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb). The A3C implementation remains largely unchanged, and serves as the based model for the A2C implementation. Julianji also has an article that explains A3C and his implementation, which was used to help assist understanding. Furthermore, Julianji also implemented this in ViZDoom. This is overall a highly recommeneded read. https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2

Additionally, Yutaro Kikuchi's work on the A3C ViZDoom project was a huge influence. Modifications to Julianjis work may be done in assistance with Yutaro's/
### Who do I talk to? ###

* Repo owner or admin
The repository creator:
	Joshua Stevens
	jd.stevens97@hotmail.com

Repository owner:
	Joshua Stevens
	jd.stevens97@hotmai.com
* Other community or team contact