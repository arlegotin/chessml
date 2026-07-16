import pytest
from torch.utils.data import DataLoader, IterableDataset

from chessml.data.iterable_dataset import ExtendedIterableDataset


class IntegersDataset(ExtendedIterableDataset):
    def __init__(self, stop=8, skip_odd=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = stop
        self.skip_odd = skip_odd

    def generator(self):
        for value in range(self.stop):
            if self.skip_odd and value % 2:
                self.skip_next()
            yield value


class MutableDataset(ExtendedIterableDataset):
    def generator(self):
        state = {"value": -1}

        for value in range(3):
            state["value"] = value
            yield state
            state["value"] = -1


def test_default_order_and_zero_finite_and_unlimited_limits():
    assert list(IntegersDataset(stop=4)) == [0, 1, 2, 3]
    assert list(IntegersDataset(stop=6, offset=2, limit=3)) == [2, 3, 4]
    assert list(IntegersDataset(stop=6, limit=0)) == []


def test_applies_transforms_left_to_right_before_offset_and_limit():
    seen = []

    def record_and_add_ten(value):
        seen.append(value)
        return value + 10

    dataset = IntegersDataset(
        stop=5,
        offset=2,
        limit=2,
        transforms=[record_and_add_ten, lambda value: value * 2],
    )

    assert list(dataset) == [24, 26]
    assert seen == [0, 1, 2, 3]


@pytest.mark.parametrize("transforms", [None, []])
def test_requires_nonempty_transforms_when_requested(transforms):
    with pytest.raises(ValueError, match="requires transforms"):
        IntegersDataset(transforms_required=True, transforms=transforms)


def test_skip_next_is_transformed_and_does_not_consume_offset_or_limit():
    seen = []

    def record(value):
        seen.append(value)
        return value

    dataset = IntegersDataset(
        stop=10,
        skip_odd=True,
        offset=1,
        limit=3,
        transforms=[record],
    )

    assert list(dataset) == [2, 4, 6]
    assert seen == list(range(7))


def test_seeded_shuffle_repeats_the_same_permutation_for_every_chunk():
    dataset = IntegersDataset(
        stop=8,
        shuffle_buffer=3,
        shuffle_seed=42,
    )
    expected = [1, 0, 2, 4, 3, 5, 7, 6]

    assert list(dataset) == expected
    assert list(dataset) == expected


def test_generated_seed_is_stable_for_full_reiteration():
    dataset = IntegersDataset(stop=8, shuffle_buffer=3)
    first = list(dataset)

    assert list(dataset) == first
    assert sorted(first) == list(range(8))


def test_transforms_snapshot_mutable_items_before_the_generator_resumes():
    dataset = MutableDataset(transforms=[lambda state: state["value"]])

    assert list(dataset) == [0, 1, 2]


def test_default_generator_reports_missing_subclass_implementation():
    with pytest.raises(NotImplementedError, match="implement generator"):
        list(ExtendedIterableDataset())


def test_remains_a_single_worker_iterable_dataset_and_leaves_batching_to_torch():
    dataset = IntegersDataset(stop=5)
    loader = DataLoader(dataset, batch_size=2, num_workers=0)

    assert isinstance(dataset, IterableDataset)
    assert [batch.tolist() for batch in loader] == [[0, 1], [2, 3], [4]]
