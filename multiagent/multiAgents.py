# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore();
        for foodPos in newFood.asList():
          score += 1.0 / util.manhattanDistance(newPos, foodPos)

        #for capsulePos in newCapsules:
        #  score += 30.0 / util.manhattanDistance(newPos, capsulePos)

        for index in range(len(newScaredTimes)):
          ghostScaredTime = newScaredTimes[index]
          ghostPos = newGhostStates[index].getPosition()

          if not ghostScaredTime:
            #print "not scared"
            if util.manhattanDistance(newPos, ghostPos) < 3:
              score -= 100

          #else:
          #  #print "scared"
          #  if ghostScaredTime < util.manhattanDistance(newPos, ghostPos):
          #    score += 30.0 / util.manhattanDistance(newPos, ghostPos)


        #print action, score
        #util.pause()
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        infinity = float('inf')

        def dispatch(gameState, depth, agentIndex):
          if agentIndex == gameState.getNumAgents(): # Check for 1 ply
            agentIndex = 0
            depth += 1

          if depth == self.depth or gameState.isWin() or gameState.isLose():
            # depth limited or terminal states reached
            return ( self.evaluationFunction(gameState), None )

          if not agentIndex:
            # pacman
            return maxValue(gameState, depth, agentIndex)

          # not pacman
          return minValue(gameState, depth, agentIndex)

        def maxValue(gameState, depth, agentIndex):
          bestValue = -infinity
          bestAction = None

          for action in gameState.getLegalActions(agentIndex):
            (value, _) = \
              dispatch(gameState.generateSuccessor(agentIndex, action), 
                       depth, agentIndex + 1)
            # determine if this value is better
            if value > bestValue:
              bestValue = value
              bestAction = action

          return (bestValue, bestAction)

        def minValue(gameState, depth, agentIndex):
          bestValue = infinity
          bestAction = None

          for action in gameState.getLegalActions(agentIndex):
            (value, _) = \
              dispatch(gameState.generateSuccessor(agentIndex, action), 
                       depth, agentIndex + 1)

            # determine if this value is better
            if value < bestValue:
              bestValue = value
              bestAction = action

          return (bestValue, bestAction)

        (bestValue, bestAction) = dispatch(gameState, 0, 0)
        #print "bestValue=", bestValue, "bestAction=", bestAction

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        infinity = float('inf')

        def dispatch(gameState, depth, agentIndex,
                     alpha = -infinity, beta = infinity):
          if agentIndex == gameState.getNumAgents(): # Check for 1 ply
            agentIndex = 0
            depth += 1

          if depth == self.depth or gameState.isWin() or gameState.isLose():
            # depth limited or terminal states reached
            return ( self.evaluationFunction(gameState), None )

          if not agentIndex:
            # pacman
            return maxValue(gameState, depth, agentIndex, alpha, beta)

          # not pacman
          return minValue(gameState, depth, agentIndex, alpha, beta)

        def maxValue(gameState, depth, agentIndex, alpha, beta):
          bestValue = -infinity
          bestAction = None

          for action in gameState.getLegalActions(agentIndex):
            (value, _) = \
              dispatch(gameState.generateSuccessor(agentIndex, action), 
                       depth, agentIndex + 1, alpha, beta)
            # determine if this value is better
            if value > bestValue:
              bestValue = value
              bestAction = action

            # update alpha and beta
            if bestValue > beta:
              return (bestValue, bestAction)
            alpha = max(alpha, bestValue)

          return (bestValue, bestAction)

        def minValue(gameState, depth, agentIndex, alpha, beta):
          bestValue = infinity
          bestAction = None

          for action in gameState.getLegalActions(agentIndex):
            (value, _) = \
              dispatch(gameState.generateSuccessor(agentIndex, action), 
                       depth, agentIndex + 1, alpha, beta)

            # determine if this value is better
            if value < bestValue:
              bestValue = value
              bestAction = action

            # update alpha and beta
            if bestValue < alpha:
              return (bestValue, bestAction)
            beta = min(beta, bestValue)

          return (bestValue, bestAction)

        (bestValue, bestAction) = dispatch(gameState, 0, 0)
        #print "bestValue=", bestValue, "bestAction=", bestAction

        return bestAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        infinity = float('inf')

        def dispatch(gameState, depth, agentIndex):
          if agentIndex == gameState.getNumAgents(): # Check for 1 ply
            agentIndex = 0
            depth += 1

          if depth == self.depth or gameState.isWin() or gameState.isLose():
            # depth limited or terminal states reached
            return ( self.evaluationFunction(gameState), None )

          if not agentIndex:
            # pacman
            return maxValue(gameState, depth, agentIndex)

          # not pacman
          return expectiValue(gameState, depth, agentIndex)

        def maxValue(gameState, depth, agentIndex):
          bestValue = -infinity
          bestAction = None

          for action in gameState.getLegalActions(agentIndex):
            (value, _) = \
              dispatch(gameState.generateSuccessor(agentIndex, action), 
                       depth, agentIndex + 1)
            # determine if this value is better
            if value > bestValue:
              bestValue = value
              bestAction = action

          return (bestValue, bestAction)

        def expectiValue(gameState, depth, agentIndex):
          totalValue = 0.0
          legalActions = gameState.getLegalActions(agentIndex)

          for action in legalActions:
            # get value of successor state
            (value, _) = \
              dispatch(gameState.generateSuccessor(agentIndex, action), 
                       depth, agentIndex + 1)

            # add up the values
            totalValue += value

          # divide value by length of legal actions to get expected value
          return (totalValue / len(legalActions), None)

        (bestValue, bestAction) = dispatch(gameState, 0, 0)
        #print "bestValue=", bestValue, "bestAction=", bestAction

        return bestAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Based on human player experience, factor in distance to 
      closest food, the ghost positions relative to pacman, and whether the 
      ghost is scared or not. 
      Pacman will always try to eat the closest food.
      Pacman will try to run away if ghost is not scared.
      Pacman will try to eat the ghost if ghost is scared.
    """
    position = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    weightFood = 10.0
    weightGhost = 1.0
    weightScaredGhost = 5.0

    score = currentGameState.getScore()

    # find the distance to the closest food
    closestDistance = float('inf')
    for food in foodList:
      distance = manhattanDistance(position, food)
      closestDistance = min(distance, closestDistance)
    score += weightFood / closestDistance

    # find the distance to the ghosts
    for ghost in ghostStates:
      distance = manhattanDistance(position, ghost.getPosition())
      if distance < 0.00000000001: continue # avoid division by zero
      #print distance

      if ghost.scaredTimer > 0:
        # ghost is scared: try to eat the ghost
        score += weightScaredGhost / distance
      else:
        # ghost is not scared: try to run away
        score -= weightGhost / distance

    return score
    
# Abbreviation
better = betterEvaluationFunction

