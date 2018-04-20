# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # Initialization
    startState = problem.getStartState()
    # print "Start:", startState

    if problem.isGoalState(startState):
        return [] # No action needed

    route = util.Stack()
    closed = set([startState])
    stack = util.Stack() # DFS use stack

    # print problem.getSuccessors(startState)
    
    for successor in problem.getSuccessors(startState):
        # Use list(old_list) to make a copy of current route
        stack.push((successor, list(route.list)))
    
    # Tree search
    while not stack.isEmpty():
        #print stack.list
        ((currentState, action, cost), route.list) = stack.pop()

        if currentState in closed:
            continue # Skip the residue of expanded states in the stack

        # print "Go ", action
        # print "In ", currentState
        route.push(action)

        if problem.isGoalState(currentState): # Check for goal condition
            # print route.list
            # util.pause()
            return route.list # Return the route
        
        # Current state is not goal state
        closed.add(currentState)
        for successor in problem.getSuccessors(currentState):
            if successor[0] in closed:
                # print "-Closed ", successor
                continue # this state is already expanded
            
            # print "-Open ", successor
            # Use list(old_list) to make a copy of current route
            stack.push((successor, list(route.list)))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Initialization
    startState = problem.getStartState()
    #print "Start:", startState

    if problem.isGoalState(startState):
        return [] # No action needed

    route = util.Stack()
    closed = set([startState])
    queue = util.Queue() # BFS use queue

    #print problem.getSuccessors(startState)
    
    for successor in problem.getSuccessors(startState):
        # Use list(old_list) to make a copy of current route
        queue.push((successor, list(route.list)))
    
    # Tree search
    while not queue.isEmpty():
        #print "Queue: ", queue.list
        ((currentState, action, cost), route.list) = queue.pop()
        
        if currentState in closed:
            continue

        #print "Go", action
        #print "In", currentState
        route.push(action)
        #print "Route", route.list

        if problem.isGoalState(currentState): # Check for goal condition
            #print ">>Finished<<", route.list
            #util.pause()
            return route.list # Return the route
        
        # Current state is not goal state
        closed.add(currentState)
        for successor in problem.getSuccessors(currentState):
            if successor[0] in closed:
                #print "-Closed ", successor
                continue # this state is already expanded
            
            #print "-Open ", successor
            # Use list(old_list) to make a copy of current route
            queue.push((successor, list(route.list)))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Initialization
    startState = problem.getStartState()

    if problem.isGoalState(startState):
        return [] # No action needed

    closedSet = set()
    queue = util.PriorityQueue()
    queue.push((startState, None, 0), 0)
    cameFrom = dict() # Stores most efficient previous action
    gScore = dict() # Stores current cost from start
    gScore[startState] = 0

    # Search
    while queue.heap: # Do while open set is not empty
        (currentState, action, cost) = queue.pop()

        if problem.isGoalState(currentState):
            # Goal reached. Construct path
            path = util.Queue() 
            
            # Backtrack to start state
            while currentState is not startState and currentState in cameFrom:
                currentState, action = cameFrom[currentState]
                path.push(action)

            return path.list

        # Expand current state
        closedSet.add(currentState) 
        for successor in problem.getSuccessors(currentState):
            successorState, successorAction, successorCost = successor
        
            if successorState in closedSet:
                continue # Skip already expanded states
            
            # Initialize entries not already in dictionaries to a big number
            if currentState not in gScore:
                gScore[currentState] = 999999999999
            if successorState not in gScore:
                gScore[successorState] = 999999999999

            # Compare this path to best path
            gTentative = gScore[currentState] + successorCost
            if gTentative >= gScore[successorState]:
                continue # Not a better path

            # A better path is found, store this path
            cameFrom[successorState] = (currentState, successorAction)
            gScore[successorState] = gTentative # Store new cost
            # Update the priority queue
            queue.update(successor, gScore[successorState]) 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Initialization
    startState = problem.getStartState()

    if problem.isGoalState(startState):
        return [] # No action needed

    closedSet = set()
    queue = util.PriorityQueue()
    queue.push((startState, None, 0), heuristic(startState, problem))
    cameFrom = dict() # Stores most efficient previous action
    gScore = dict() # Stores current cost from start
    gScore[startState] = 0

    # Search
    while queue.heap: # Do while open set is not empty
        (currentState, action, cost) = queue.pop()

        if problem.isGoalState(currentState):
            # Goal reached. Construct path
            path = util.Queue()

            # Backtrack to start state
            while currentState is not startState and currentState in cameFrom:
                currentState, action = cameFrom[currentState]
                path.push(action)

            return path.list

        # Expand current state
        closedSet.add(currentState)
        for successor in problem.getSuccessors(currentState):
            successorState, successorAction, successorCost = successor
        
            if successorState in closedSet:
                continue # Skip expanded states

            # Initialize entries not already in dictionaries to a big number
            if currentState not in gScore:
                gScore[currentState] = 999999999999
            if successorState not in gScore:
                gScore[successorState] = 999999999999

            # Compare this path to best path
            gTentative = gScore[currentState] + successorCost
            if gTentative >= gScore[successorState]:
                continue # Not a better path

            # A better path is found, store this path
            cameFrom[successorState] = (currentState, successorAction)
            gScore[successorState] = gTentative # Store new cost

            # Update priority queue with new heuristic estimate
            queue.update(successor, (gScore[successorState]
                                     + heuristic(successorState, problem)))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
