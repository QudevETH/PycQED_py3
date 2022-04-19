STATE_ORDER = "gefhabcdijklmnopqrtuvwxyz0123456789"


def transmon_state_to_str(state):
    """
    Translates the transmon state to a one-letter string. Acts as identity if
    state is already given as one-letter string

    Arguments:
        Transmon state either as 1-string (character) or integer

    Returns:
        Transmon state as a single character (i.e. string of length 1)

    Examples::
        0 -> 'g'
        1 -> 'e'
      'h' -> 'h'
    """
    if type(state) == int:
        if 0 <= state <= 34:
            return STATE_ORDER[state]
    elif type(state) == str:
        if len(state) == 1:
            return state
    raise NotImplementedError(
        f"State not recognized. Function only accepts numbers from 0 until 34 "
        f"or characters in {STATE_ORDER}")


def transmon_state_to_int(state):
    """
    Translates the transmon state to an integer. Leaves integers between 0 and
    34 as is.

    Arguments:
        Transmon state either as a string of a single character or integer

    Returns:
        Transmon state as integer

    Examples::
        'g' -> 0
        'e' -> 1
    """
    if type(state) == int:
        if 0 <= state <= 34:
            return state
    elif type(state) == str:
        return STATE_ORDER.index(state)
    raise NotImplementedError(
        f"State not recognized. Function only accepts numbers from 0 until 34 "
        f"or characters in {STATE_ORDER}")


def transmon_resonator_state_to_str(state):
    """
    Translates the transmon-resonator state to string. Leaves states that are
    correctly formatted as string as is.

    Arguments:
        Transmon-resontator state either as string or tuple

    Returns:
        Transmon-resonator state as string

    Examples::
        (0,3) -> 'g,3'
        (1,0) -> 'e,0'
        'h,3' -> 'h,3'
    """
    if type(state) == tuple:
        return f"{transmon_state_to_str(state[0])},{str(state[1])}"
    elif type(state) == str:
        if len(state) == 1:
            return f'{state},0'
        else:
            return state
    raise NotImplementedError


def transmon_resonator_state_to_tuple(state):
    """
    Translates the transmon-resonator state to tuple.

    Arguments:
        Transmon-resontator state either as string or tuple

    Returns:
        Transmon-resonator state as 2-tuple of integers

    Examples::
        'g,3' -> (0,3)
        'e,0' -> (1,0)
        (2,2) -> (2,2)
    """
    if type(state) == tuple:
        return state
    elif type(state) == str:
        state_lst = state.split(",")
        return (transmon_state_to_int(state_lst[0]), int(state_lst[1]))
    raise NotImplementedError


def transition_to_str(transition):
    """
    Translates transitions represented as tuples to string representation
    which is more readable.
    The function can also handle strings (it will function as the identity map
    in that case).

    Parameters:
        transition: transition either as
            - tuple. In this case the tuple representation of the transition
            given as string will be returned. See
            examples below.
            - string. In this case simply the same string will be returned.

    Returns:
        Tuple representation of the transition

    Examples::
        ((0,0),(1,2)) -> 'g,0-e,2'
        ((1,0),(2,0)) -> 'e,0-f,0'
        'h,3-f,0' -> 'h,3-f,0'

    """
    if type(transition) == str:
        return transition
    elif type(transition) == tuple:
        if transition[0][1] == 0 and transition[1][1] == 0:
            return STATE_ORDER[transition[0][0]] + STATE_ORDER[transition[
                1][0]]
        else:
            return f"{transmon_state_to_str(transition[0][0])}," \
                   f"{str(transition[0][1])}-" \
                   f"{transmon_state_to_str(transition[1][0])}," \
                   f"{str(transition[1][1])}"


def transition_to_tuple(transition):
    """
    Translates transitions represented as strings to tuple representation.
    The function can also handle tuples (it will function as the identity map
    in that case).

    Parameters:
        transition: transition either as
            - string formatted like transmon state, resonator state-transmon.
            For example: 'g,0-e,2'
            - common transitions like 'ge' and 'ef' will return ((0,0),(1,
            0)) and ((1,0),(2,0)). Note that the resonator
            is assumed to be in the ground state for these transitions.
            - tuple. This will just return the same tuple.

    Returns:
        Tuple representation of the transition

    Examples::
        'g,0-e,2' -> ((0,0),(1,2))
        'ef' -> ((1,0),(2,0))
        ((0,0),(1,2)) -> ((0,0),(1,2))

    """
    if type(transition) == tuple:
        return transition
    elif type(transition) == str:
        if len(transition) == 2:
            return ((STATE_ORDER.index(transition[0]), 0),
                    (STATE_ORDER.index(transition[1]), 0))
        else:
            transition = tuple(
                [tuple(x.split(',')) for x in transition.split('-')])
            return (transmon_state_to_int(transition[0][0]),
                    int(transition[0][1])), \
                   (transmon_state_to_int(transition[1][0]),
                    int(transition[1][1]))
