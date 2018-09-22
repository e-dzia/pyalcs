from lcs.agents.acs2 import ClassifiersList
from lcs.strategies.action_planning.goal_sequence_searcher \
    import GoalSequenceSearcher


def exists_classifier(classifiers, previous_situation, action, situation,
                      quality):
    """
    Returns True if there is a classifier in this list with a quality
    higher than 'quality' that matches previous_situation,
    specifies action, and predicts situation.
    Returns False otherwise.
    :param classifiers:
    :param previous_situation:
    :param action:
    :param situation:
    :param quality:
    :return:
    """
    for cl in classifiers:
        if cl.q > quality and cl.does_match(previous_situation) \
            and cl.action == action \
            and cl.does_anticipate_correctly(previous_situation,
                                             situation):
            return True
    return False


def get_quality_classifiers_list(classifiers, quality, cfg=None):
    """
    Constructs classifier list out of a list with q > quality.
    :param classifiers:
    :param quality:
    :param cfg:
    :return: ClassifiersList with only quality classifiers.
    """
    listp = ClassifiersList(cfg=cfg)
    for item in classifiers:
        if item.q > quality:
            listp.append(item)
    return listp


def search_goal_sequence(classifiers, start, goal):
    """
    Searches a path from start to goal using a bidirectional method in the
    environmental model (i.e. the list of reliable classifiers).
    :param start: Perception
    :param goal: Perception
    :return: Sequence of actions
    """
    reliable_classifiers = \
        get_quality_classifiers_list(classifiers,
                                     quality=classifiers.cfg.theta_r,
                                     cfg=classifiers.cfg)

    return GoalSequenceSearcher().search_goal_sequence(
        reliable_classifiers, start, goal)
