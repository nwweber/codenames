2021-11-14
* good resource for how to think about this problem: https://towardsdatascience.com/training-an-ai-to-play-warhammer-40k-part-one-planning-78aa5dfa888a
* practical starter: https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
* above implements game as openai gym environment: https://gym.openai.com/
* would be good starting point to do that
* then can apply out of the box RL algos: https://stable-baselines3.readthedocs.io/en/master/
* my task would be to craft environment: state representation, rewards, and (implicitly) game rules
* big issue: codenames is multiplayer, thus also multi-agent, game
* gym equivalent for multi agent environments: https://github.com/Farama-Foundation/PettingZoo
* unclear which learning algos would work with that, needs research
* multi agent is huge complication! need to decide way forward
  * single agent: RL agent guesser, vs heuristics/hard-coded hint giver. guesser much more constrained, easier to
  learn and implement than hint giver
  * single agent: RL agent hint giver, heuristic guesser. Also cool?
  * multi agent: the actual thing. let agents learn to play together (guesser and hint giver) and against one another
  (across teams)
* additional concern: multi-agent petting zoo environment supports 'legal actions' for given state. pretty important 
for this game, otherwise lots of learning time wasted on that. hmm
* can i simplify codenames to be single player game (essentially)?
* consider implementing duet instead? https://www.youtube.com/watch?v=PUTNDlnxLk8