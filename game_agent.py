"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    w, h = game.width, game.height
    y, x = game.get_player_location(player)
    center_score = float((h - y) ** 2 + (w - x) ** 2)

    return float(own_moves - opp_moves) * (center_score)**0.5



def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    w, h = game.width, game.height
    y, x = game.get_player_location(player)
    center_score = float((h - y) ** 2 + (w - x) ** 2)

    return float(own_moves - opp_moves) * (center_score)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))

    w, h = game.width, game.height
    y, x = game.get_player_location(player)
    center_score = float((h - y) ** 2 + (w - x) ** 2)

    return float(own_moves) * (center_score)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    # TODO: finish this function!
    def scores_depth_limited(self, game, depth):
        """
        Find the scores for all the possible moves down to the given depth

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        scores: dictionary
            scores for all the possible board states down to the depth
        """

        # Used for BFS search for each exploring children of each node
        queue = [['0', game]]

        # For capturing the scores
        scores_dict = dict()

        # Will be used for next_best_move function
        scores_dict['depth'] = depth
        scores_dict['width'] = game.width
        scores_dict['height'] = game.height

        # Top node is set to -inf in case there are no legal moves
        scores_dict['0'] = [float('-inf')]

        # Initial blank spaces used for finding the current level (depth)
        initial_blank_spaces = game.get_blank_spaces()

        while queue:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # BFS so first in - first out and therefore using the first element
            l, g = queue.pop(0)

            # Finding the level explored so far based on the number of blank spaces
            blank_spaces = g.get_blank_spaces()
            level = len(initial_blank_spaces) - len(blank_spaces)

            # Possible moves are basically overlap of blank spaces and the possible moved
            possible_moves = g.get_legal_moves()
            possible_moves = list(set(possible_moves) & set(blank_spaces))

            # If reached the bottom level, just return the scores -- Nothing else to do
            if level == (depth - 1):
                for i in range(len(possible_moves)):
                    # for move in possible_moves:
                    new_board = g.forecast_move(possible_moves[i])
                    scores_dict[str(l) + '-' + str(i)] = [self.score(new_board, game._player_1), possible_moves[i]]

            else:
                for i in range(len(possible_moves)):
                    # for move in possible_moves:
                    new_board = g.forecast_move(possible_moves[i])
                    queue.append([str(l) + '-' + str(i), new_board])
                    scores_dict[str(l) + '-' + str(i)] = [None, possible_moves[i]]

        return scores_dict


    def next_best_move(self, scores):
        """
        Find the next best move using miniMax Logic
        Parameters
        ----------
        scores: dictionary 'board state key': {'score', 'move'}
            scores for all the possible board states down to the depth.
            It also contains the depth explored.
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        width = scores['width']
        height = scores['height']

        # MiniMax starts with the level right before the bottom level
        level = scores['depth'] - 1

        del scores['depth'], scores['width'], scores['height']

        # Continue until we reach the top node
        while level >= 0:

            # Find all the nodes in the target level
            # scores keys are constructed as level0-level1-level2-...-leveln
            nodes = [x for x in scores if (len(x.split('-')) - 1) == level]

            if level == 0:

                best_move = (-1, -1)

                for node in nodes:
                    # -inf for MAX (inf for Min) If there are no legal moves
                    max_score = float('-inf')

                    # The branching factor is definitely less than the board size!
                    try:
                        # Go over all childs
                        for i in range(width * height):

                            # Find the child score
                            child_node = node + '-' + str(i)
                            child_score = scores[child_node][0]

                            # If found a new MAX
                            if child_score > max_score:
                                max_score = max(child_score, max_score)
                                # Find the best move so far
                                best_move = scores[child_node][-1]
                    except:
                        continue

                return best_move




            elif level % 2 == 0:

                for node in nodes:

                    # -inf for MAX (inf for Min) If there are no legal moves
                    max_score = float('-inf')

                    # The branching factor is definitely less than the board size!
                    try:

                        # Go over all childs
                        for i in range(width * height):

                            # Find the child score
                            child_node = node + '-' + str(i)
                            child_score = scores[child_node][0]

                            # If found a new MAX
                            if child_score > max_score:
                                max_score = max(child_score, max_score)

                                # Updated the score with the new MAX
                                scores[node][0] = max_score  # the same as child_score

                    except:

                        continue



            else:

                for node in nodes:

                    # -inf for MAX (inf for Min) If there are no legal moves
                    min_score = float('inf')

                    # The branching factor is definitely less than the board size!
                    try:

                        # Go over all children
                        for i in range(width * height):

                            # Find the child score
                            child_node = node + '-' + str(i)
                            child_score = scores[child_node][0]

                            # If found a new MAX
                            if child_score < min_score:
                                min_score = min(child_score, min_score)

                                # Updated the score with the new MAX
                                scores[node][0] = min_score  # the same as child_score


                    except:

                        continue

            # Move up on level and use the same logic -- MiniMax
            level -= 1


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        scores = self.scores_depth_limited(game, depth)
        best_move = self.next_best_move(scores)

        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left, alpha=float("-inf"), beta=float("inf")):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        initial_depth = 1

        # TODO: finish this function!
        while True:

            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move = self.alphabeta(game, initial_depth, alpha, beta)

            except SearchTimeout:
                # Handle any actions required after timeout as needed
                # Return the best move from the last completed search iteration
                return best_move

            initial_depth += 1


    def scores_depth_limited_alphaBeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """
        Find the scores for all the possible moves down to the given depth

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        scores: dictionary
            scores for all the possible board states down to the depth.
            The value of the dictionary: [score, move, , alpha, beta, parent_node_id]
        """

        # Using BFS for exploring children of each node
        queue = [['0', game]]

        # For capturing the scores
        scores_dict = dict()

        # Used for setting the unexplored node to their worse possible value
        inf = float('inf')

        # Initial blank spaces used for finding the current level (depth)
        initial_blank_spaces = game.get_blank_spaces()

        # Top node score and move are set to -inf and (-1,-1), respectively,
        # in case there are no legal moves
        scores_dict['0'] = {'score': -inf, 'move': (-1, -1), 'alpha': alpha,
                            'beta': beta, 'parent': None, 'level': 0, 'children': []}

        while queue:

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # BFS so first in - first out and therefore using the first element
            # l:level & g:boardState (game)
            l, g = queue.pop(0)

            # Finding the level explored so far based on the number of blank spaces
            blank_spaces = g.get_blank_spaces()
            level = len(initial_blank_spaces) - len(blank_spaces)

            # Possible moves are basically overlap of blank spaces and the possible moved
            possible_moves = g.get_legal_moves()
            possible_moves = list(set(possible_moves) & set(blank_spaces))

            # If reached the bottom level, just return the scores -- Do not add to the queue.
            if level == (depth - 1):

                # for move in possible_moves:
                for i in range(len(possible_moves)):
                    # Child node
                    child_node = possible_moves[i]

                    # Adding to the list of the children
                    scores_dict[l]['children'].append(str(l) + '-' + str(i))

                    # Finding the state of the board after applyting the possible move
                    new_board = g.forecast_move(child_node)

                    # Including them in the dictionary
                    scores_dict[str(l) + '-' + str(i)] = {'score': self.score(new_board, game._player_1),
                                                          'move': child_node, 'alpha': alpha,
                                                          'beta': beta, 'parent': l,
                                                          'level': level + 1, 'children': []}

            # Otherwise, add them to do the queue
            else:

                # for move in possible_moves:
                for i in range(len(possible_moves)):

                    # Child node
                    child_node = possible_moves[i]

                    # Adding to the list of the children
                    scores_dict[l]['children'].append(str(l) + '-' + str(i))

                    # Finding the state of the board after applyting the possible move
                    new_board = g.forecast_move(child_node)

                    # Adding the next board to the queue to find its children
                    queue.append([str(l) + '-' + str(i), new_board])

                    # Including them in the dictionary
                    # Although there exist values only for the bottom level nodes
                    # all nodes need to be included for find the next_best_move

                    if level % 2 == 0:  # Minimizer
                        scores_dict[str(l) + '-' + str(i)] = {'score': inf, 'move': child_node,
                                                              'alpha': alpha, 'beta': beta, 'parent': l,
                                                              'level': level + 1, 'children': []}
                    else:  # Maximizer
                        scores_dict[str(l) + '-' + str(i)] = {'score': -inf, 'move': child_node,
                                                              'alpha': alpha, 'beta': beta, 'parent': l,
                                                              'level': level + 1, 'children': []}

        return scores_dict



    def next_best_move_alphaBeta(self, input_scores, stack=['0'], depth=3):
        """
        Find the next best move using miniMax Logic
        Parameters
        ----------
        scores: dictionary 'board state key': {'score', 'move'}
            scores for all the possible board states down to the depth.
            It also contains the depth explored.
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Return the scores after exhausting all the options return the updated scores
        if not stack:
            return input_scores['0']['move']

        else:

            node = stack.pop()

            # For top node, there is no parent
            # And it is just the matter of capturing the children nodes
            # For the following DFS
            if input_scores[node]['level'] == 0:

                # Adding all the children to the stack
                for child_node in input_scores[node]['children'][::-1]:
                    # Add them to the stack as this is a DFS
                    stack.append(child_node)

                return self.next_best_move_alphaBeta(input_scores, stack=stack)


            # If a MiniMizer
            elif (input_scores[node]['level'] % 2 == 1) and (input_scores[node]['level'] != depth):

                # Finding the parent node, in case need to traverse back up
                # And inheriting the Beta/Alpha
                parent_node = input_scores[node]['parent']

                # And remove the child from the parents list of children (to be explored)
                # It is important to do so, as only when all the children are
                # explored, updating parents alpha/beta can be done.
                input_scores[parent_node]['children'].remove(node)

                # Alpha and Beta are always inherited from the parent node
                input_scores[node]['alpha'] = input_scores[parent_node]['alpha']
                input_scores[node]['beta'] = input_scores[parent_node]['beta']

                # Flag for pruning. If not pruning, traverse up the tree
                pruned = False

                # Adding all the children to the stack
                for child_node in input_scores[node]['children'][::-1]:

                    # Add them to the stack as this is a DFS
                    stack.append(child_node)

                    # If these condition is not met --> Pruning can take place
                    if input_scores[node]['beta'] > input_scores[node]['alpha']:

                        # Only if the children have a value other than the
                        # initial given values, the node values can get updated
                        if input_scores[child_node]['score'] != -float('inf'):
                            # Update Node's Score & Beta before proceeding
                            input_scores[node]['score'] = min(input_scores[node]['score'],
                                                              input_scores[child_node]['score'])
                            input_scores[node]['beta'] = min(input_scores[node]['beta'],
                                                             input_scores[node]['score'])

                            # And remove the child from the parents list of children (to be explored)
                            # It is important to do so, as only when all the children are
                            # explored, updating parents alpha/beta can be done.
                            input_scores[node]['children'].remove(child_node)

                    else:
                        pruned = True
                        input_scores[node]['children'].remove(child_node)


                # Updates all the parent nodes along the path to the root node
                if input_scores[node]['score'] != float('inf'):

                    while (len(node) > 1) and (not input_scores[node]['children']):

                        # Minimizer
                        if len(node.split('-')) % 2 == 0:

                            parent_alpha = max(input_scores[node]['beta'],
                                               input_scores[parent_node]['alpha'])
                            input_scores[parent_node]['alpha'] = parent_alpha

                            parent_score = max(input_scores[node]['score'],
                                               input_scores[parent_node]['score'])

                            # If it is the top node, inheret the move, as well!
                            if parent_node == '0' and float(input_scores[node]['score']) >= float(input_scores[parent_node]['score']):

                                input_scores[parent_node]['move'] = input_scores[node]['move']

                            input_scores[parent_node]['score'] = parent_score

                        # Maximizer
                        else:

                            parent_beta = min(input_scores[node]['alpha'],
                                              input_scores[parent_node]['beta'])
                            input_scores[parent_node]['beta'] = parent_beta

                            parent_score = min(input_scores[node]['score'],
                                               input_scores[parent_node]['score'])
                            input_scores[parent_node]['score'] = parent_score

                        # Going a level higher
                        node, parent_node = parent_node, parent_node[:-2]

                return self.next_best_move_alphaBeta(input_scores, stack=stack)



            # If a MAXIMIZER
            elif (input_scores[node]['level'] % 2 == 0) and (input_scores[node]['level'] != depth):

                # Finding the parent node, in case need to traverse back up
                # And inheriting the Beta/Alpha
                parent_node = input_scores[node]['parent']

                # And remove the child from the parents list of children (to be explored)
                # It is important to do so, as only when all the children are
                # explored, updating parents alpha/beta can be done.
                input_scores[parent_node]['children'].remove(node)

                # Alpha and Beta are always inherited from the parent node
                input_scores[node]['alpha'] = input_scores[parent_node]['alpha']
                input_scores[node]['beta'] = input_scores[parent_node]['beta']

                # Flag for pruning. If not pruning, traverse up the tree
                pruned = False

                # Adding all the children to the stack
                for child_node in input_scores[node]['children'][::-1]:

                    # Add them to the stack as this is a DFS
                    stack.append(child_node)

                    # If these condition is not met --> Pruning can take place
                    if input_scores[node]['beta'] > input_scores[node]['alpha']:

                        # Only if the children have a value other than the
                        # initial given values, the node values can get updated
                        if input_scores[child_node]['score'] != float('inf'):
                            # Update Node's Score & ALPHA before proceeding
                            input_scores[node]['score'] = max(input_scores[node]['score'],
                                                              input_scores[child_node]['score'])
                            input_scores[node]['alpha'] = max(input_scores[node]['alpha'],
                                                              input_scores[node]['score'])

                            # And remove the child from the parents list of children (to be explored)
                            # It is important to do so, as only when all the children are
                            # explored, updating parents alpha/beta can be done.
                            input_scores[node]['children'].remove(child_node)

                    else:
                        pruned = True
                        input_scores[node]['children'].remove(child_node)


                # Updates all the parent nodes along the path to the root node
                if input_scores[node]['score'] != -float('inf'):

                    while (len(node) > 1) and (not input_scores[node]['children']):

                        # Minimizer
                        if len(node.split('-')) % 2 == 0:

                            parent_alpha = max(input_scores[node]['beta'],
                                               input_scores[parent_node]['alpha'])
                            input_scores[parent_node]['alpha'] = parent_alpha

                            parent_score = max(input_scores[node]['score'],
                                               input_scores[parent_node]['score'])

                            # If it is the top node, inheret the move, as well!
                            if parent_node == '0' and float(input_scores[node]['score']) >= float(input_scores[parent_node]['score']):

                                input_scores[parent_node]['move'] = input_scores[node]['move']

                            input_scores[parent_node]['score'] = parent_score

                        # Maximizer
                        else:

                            parent_beta = min(input_scores[node]['alpha'],
                                              input_scores[parent_node]['beta'])
                            input_scores[parent_node]['beta'] = parent_beta

                            parent_score = min(input_scores[node]['score'],
                                               input_scores[parent_node]['score'])
                            input_scores[parent_node]['score'] = parent_score

                        # Going a level higher
                        node, parent_node = parent_node, parent_node[:-2]

                return self.next_best_move_alphaBeta(input_scores, stack=stack)


            # In the case it is the bottom level children, just continue
            else:
                return self.next_best_move_alphaBeta(input_scores, stack=stack)



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        scores = self.scores_depth_limited_alphaBeta(game, depth, alpha=alpha, beta=beta)
        best_move = self.next_best_move_alphaBeta(scores, depth=depth)

        return best_move
