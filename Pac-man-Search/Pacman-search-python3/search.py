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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    '''The logic behind this is to use stack to record the latest node we visited, and pop it if its has no new nodes
    next to it (all visited, in a corner...)'''
    fringe = util.Stack()
    '''where to start, type: (x, y)'''
    location = problem.getStartState()
    '''moves is a list of buttons to push to get to the goal, East, West, North, South'''
    moves = []
    '''a list of locations visited'''
    # visited_locations = [location]
    # '''push the original node into stack, data structure: (where I am, how did I get here, where have I visited)'''
    # fringe.push((location, moves, visited_locations))
    #
    # # if stack is not empty keep going
    # while not fringe.isEmpty():
    #     # latest node popped and checked
    #     location, moves, visited_locations = fringe.pop()
    #     # if reached the goal return the latest synced path.
    #     if problem.isGoalState(location):
    #         return moves
    #
    #     # else get the successor of the node, skip if no successor
    #     for next_node, direction, cost in problem.getSuccessors(location):
    #         # if new land discovered, push to stack, explore first later.
    #         if next_node not in visited_locations:
    #             # again, (where I am, how did I get here, where have I visited)
    #             fringe.push((next_node, moves + [direction], visited_locations + [location]))
    # # if everything is popped means it has no path to goal.
    # return []
    '''The logic behind this is to use stack to record the latest node we visited, and pop it if its has no new nodes
        next to it (all visited, in a corner...)'''
    fringe = util.Stack()
    '''where to start, type: (x, y)'''
    location = problem.getStartState()
    '''moves is a list of buttons to push to get to the goal, East, West, North, South'''
    moves = []
    visited_locations = []
    '''push the original node into stack, data structure: (where I am, how did I get here)'''
    fringe.push((location, moves))
    while not fringe.isEmpty():
        # latest node popped and checked
        location, moves = fringe.pop()
        # if goal reached return all moves
        if problem.isGoalState(location):
            return moves
        # if node was not visited
        if location not in visited_locations:
            # stamp on it
            visited_locations.append(location)
            # find next positions
            for next_location, direction, cost in problem.getSuccessors(location):
                fringe.push((next_location, moves + [direction]))
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    '''as DFS but use queue to pop the first node in the pile'''
    fringe = util.Queue()
    '''where to start, type: (x, y)'''
    location = problem.getStartState()
    '''moves is a list of buttons to push to get to the goal, East, West, North, South'''
    moves = []
    visited_locations = []
    '''push the original node into stack, data structure: (where I am, how did I get here)'''
    fringe.push((location, moves))
    # if stack is not empty keep going
    while not fringe.isEmpty():
        # first node popped and checked
        location, moves = fringe.pop()
        if problem.isGoalState(location):
            return moves
        if location not in visited_locations:
            visited_locations.append(location)
            for next_location, direction, cost in problem.getSuccessors(location):
                fringe.push((next_location, moves + [direction]))
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    '''where to start, type: (x, y)'''
    location = problem.getStartState()
    '''moves is a list of buttons to push to get to the goal, East, West, North, South'''
    moves = []
    cost = 0
    visited_locations = []
    '''push the original node into stack, data structure: (where I am, how did I get here)'''
    fringe.push((location, moves, cost), 0)
    while not fringe.isEmpty():
        # latest node popped and checked
        location, moves, cost = fringe.pop()
        # if goal reached return all moves
        if problem.isGoalState(location):
            return moves
        # if node was not visited
        if location not in visited_locations:
            # stamp on it
            visited_locations.append(location)
            # find next positions
            for next_location, direction, current_cost in problem.getSuccessors(location):
                fringe.push((next_location, moves + [direction], cost + current_cost), cost + current_cost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
