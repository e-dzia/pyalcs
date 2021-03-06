from random import randint

import pytest

from lcs import Perception
from lcs.agents.acs2 import Configuration, ClassifiersList, Classifier
from lcs.strategies.action_planning.action_planning import \
    get_quality_classifiers_list, exists_classifier, search_goal_sequence


class TestActionPlanning:

    @pytest.fixture
    def cfg(self):
        return Configuration(8, 8, theta_r=0.9)

    def test_get_quality_classifiers_list(self, cfg):
        # given
        population = ClassifiersList()

        # C1 - matching
        c1 = Classifier(quality=0.9, cfg=cfg)

        # C2 - matching
        c2 = Classifier(quality=0.7, cfg=cfg)

        # C3 - non-matching
        c3 = Classifier(quality=0.5, cfg=cfg)

        # C4 - non-matching
        c4 = Classifier(quality=0.1, cfg=cfg)

        population.append(c1)
        population.append(c2)
        population.append(c3)
        population.append(c4)

        # when
        match_set = get_quality_classifiers_list(population, 0.5)

        # then
        assert 2 == len(match_set)
        assert c1 in match_set
        assert c2 in match_set

    def test_exists_classifier(self, cfg):
        # given
        population = ClassifiersList()
        prev_situation = Perception('01100000')
        situation = Perception('11110000')
        act = 0
        q = 0.5

        # C1 - OK
        c1 = Classifier(condition='0##0####', action=0, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C2 - wrong action
        c2 = Classifier(condition='0##0####', action=1, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C3 - wrong condition
        c3 = Classifier(condition='0##1####', action=0, effect='1##1####',
                        quality=0.7, cfg=cfg)

        # C4 - wrong effect
        c4 = Classifier(condition='0##0####', action=0, effect='1##0####',
                        quality=0.7, cfg=cfg)

        # C5 - wrong quality
        c5 = Classifier(condition='0##0####', action=0, effect='1##1####',
                        quality=0.25, cfg=cfg)

        population.append(c2)
        population.append(c3)
        population.append(c4)
        population.append(c5)

        # when
        result0 = exists_classifier(population,
                                    p0=prev_situation,
                                    p1=situation, action=act, quality=q)

        population.append(c1)
        result1 = exists_classifier(population,
                                    p0=prev_situation,
                                    p1=situation, action=act, quality=q)

        # then
        assert result0 is False
        assert result1 is True

    def test_search_goal_sequence_1(self, cfg):
        # given
        start = "01111111"
        goal = "00111111"

        classifiers = ClassifiersList(
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.88, cfg=cfg),
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.92, cfg=cfg)
        )

        # when
        result = search_goal_sequence(classifiers, start, goal, cfg.theta_r)

        # then
        assert result == [1]

    def test_search_goal_sequence_2(self, cfg):
        # given
        start = "01111111"
        goal = "00111111"

        classifiers = ClassifiersList(
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.88, cfg=cfg),
            Classifier(condition="#0######", action=1, effect="#1######",
                       quality=0.98, cfg=cfg)
        )

        # when
        result = search_goal_sequence(classifiers, start, goal, cfg.theta_r)

        # then
        assert result == []

    def test_search_goal_sequence_3(self, cfg):
        # given
        start = "01111111"
        goal = "10111111"

        classifiers = ClassifiersList(
            Classifier(condition="#1######", action=1, effect="#0######",
                       quality=0.94, cfg=cfg),
            Classifier(condition="0#######", action=2, effect="1#######",
                       quality=0.98, cfg=cfg),
        )

        # when
        result = search_goal_sequence(classifiers, start, goal, cfg.theta_r)

        # then
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
