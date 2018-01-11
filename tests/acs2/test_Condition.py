import pytest

from alcs.acs2 import ACS2Configuration, Condition

from alcs import Perception
from .randommock import SampleMock


class TestCondition:

    @pytest.fixture
    def cfg(self):
        return ACS2Configuration(8, 8)

    def test_equal(self, cfg):
        assert Condition('########', cfg=cfg) == Condition('########', cfg=cfg)
        assert Condition('1#######', cfg=cfg) != Condition('########', cfg=cfg)
        assert Condition('########', cfg=cfg) != Condition('#######1', cfg=cfg)
        assert Condition('1111####', cfg=cfg) == Condition('1111####', cfg=cfg)
        assert Condition('1111####', cfg=cfg) != Condition('1011####', cfg=cfg)
        assert Condition('1101####', cfg=cfg) != Condition('1111####', cfg=cfg)
        assert Condition('00001###', cfg=cfg) == Condition('00001###', cfg=cfg)

    def test_should_generalize(self, cfg):
        # given
        s = "#1O##O##"
        cond = Condition(s, cfg=cfg)

        # when
        cond.generalize(position=1)

        # then
        assert Condition("##O##O##", cfg=cfg) == cond

    def test_generalize_decrements_specificity(self, cfg):
        # given
        condition = Condition('#11#####', cfg=cfg)
        assert 2 == condition.specificity

        # when
        condition.generalize(1)

        # then
        assert 1 == condition.specificity

    def test_should_only_accept_strings(self, cfg):
        condition = Condition(cfg=cfg)

        with pytest.raises(TypeError):
            # Try to store an integer
            condition[0] = 1

    def test_should_initialize_two_times_the_same_way(self, cfg):
        # given
        c1 = Condition("#1O##O##", cfg=cfg)
        c2 = Condition(['#', '1', 'O', '#', '#', 'O', '#', '#'], cfg=cfg)

        # then
        assert c1 == c2

    def test_should_return_number_of_specified_elements(self, cfg):
        # given
        condition = Condition(cfg=cfg)
        assert 0 == condition.specificity

        # when
        condition.specialize(2, '1')
        condition.specialize(5, '0')

        # then
        assert 2 == condition.specificity

    def test_should_get_initialized_with_str_1(self, cfg):
        # given
        condition = Condition("#1O##O##", cfg=cfg)

        # then
        assert 8 == len(condition)

    def test_should_get_initialized_with_str_2(self, cfg):
        with pytest.raises(ValueError):
            # Too short condition
            Condition("#1O##O#", cfg=cfg)

    def test_should_specialize_1(self, cfg):
        # given
        cond = Condition(cfg=cfg)
        diff = Condition('#0###1#1', cfg=cfg)

        # when
        cond.specialize(new_condition=diff)

        # then
        assert Condition('#0###1#1', cfg=cfg) == cond

    def test_should_specialize_2(self, cfg):
        # given
        c = Condition('###10#1#', cfg=cfg)
        diff = Condition('010##1##', cfg=cfg)

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition('0101011#', cfg=cfg) == c

    def test_should_specialize_3(self, cfg):
        # given
        c = Condition('#101#10#', cfg=cfg)
        diff = Condition('####1##1', cfg=cfg)

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition('#1011101', cfg=cfg) == c

    def test_should_specialize_4(self, cfg):
        # given
        c = Condition('####01#1', cfg=cfg)
        diff = Condition('2#00####', cfg=cfg)

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition(['2', '#', '0', '0', '0', '1', '#', '1'], cfg=cfg) == c

    def test_should_specialize_5(self, cfg):
        # given
        c = Condition(['#', '#', '#', '0', '1', '#', '0', '#'], cfg=cfg)
        diff = Condition(['1', '0', '1', '#', '#', '0', '#', '#'], cfg=cfg)

        # when
        c.specialize(new_condition=diff)

        # then
        assert Condition(['1', '0', '1', '0', '1', '0', '0', '#'], cfg=cfg) == c

    def test_should_match_perception(self, cfg):
        # given
        c = Condition(cfg=cfg)
        p = Perception(['1', '0', '0', '1', '1', '0', '0', '1'])

        # General condition - should match everything
        assert c.does_match(p) is True

        # Correct first position
        c.specialize(0, '1')
        assert c.does_match(p) is True

        # Expects 0 as the first condition
        c.specialize(0, '0')
        assert c.does_match(p) is False

        # Should fail when perception length is different
        with pytest.raises(ValueError):
            c.does_match(Perception(['1', '2']))

    def test_should_match_condition_1(self, cfg):
        c_empty = Condition(cfg=cfg)
        c = Condition(['1', '0', '0', '1', '1', '0', '0', '1'], cfg=cfg)

        # General condition - should match everything
        assert c_empty.does_match(c) is True

        # Correct first position
        c_empty.specialize(0, '1')
        assert c_empty.does_match(c) is True

        # Expects 0 as the first condition
        c_empty.specialize(0, '0')
        assert c_empty.does_match(c) is False

    def test_should_match_condition_2(self, cfg):
        # Given
        c = Condition('####O###', cfg=cfg)
        other = Condition('#1O##O##', cfg=cfg)

        # When
        res = c.does_match(other)

        # Then
        assert res is True

    def test_crossover(self, cfg):
        # given
        c1 = Condition('0##10###', cfg=cfg)
        c2 = Condition('#10##0##', cfg=cfg)

        # when
        c1.two_point_crossover(c2, samplefunc=SampleMock([1, 4]))

        # then
        assert Condition('010#0###', cfg=cfg) == c1
        assert Condition('###1#0##', cfg=cfg) == c2

    def test_crossover_allows_to_change_last_element(self, cfg):
        # given
        c1 = Condition('0##10###', cfg=cfg)
        c2 = Condition('#10##011', cfg=cfg)

        # when
        c1.two_point_crossover(c2, samplefunc=SampleMock([5, 8]))

        # then
        assert Condition('0##10011', cfg=cfg) == c1
        assert Condition('#10#####', cfg=cfg) == c2
