import hnsw_cpp
import pytest

def test_construct_index():
    index = hnsw_cpp.HNSWIndex(3, 16, 64)
    assert index is not None


def test_add_and_search():
    index = hnsw_cpp.HNSWIndex(3, 16, 64)

    index.add([1.0, 2.0, 3.0])
    index.add([10.0, 10.0, 10.0])
    index.add([1.1, 2.1, 3.1])

    results = index.search(
        [1.0, 2.0, 3.0],
        2,
        10
    )

    assert len(results) == 2
    assert results[0][0] == 0


def test_result_shape():
    index = hnsw_cpp.HNSWIndex(2, 8, 32)

    index.add([0.0, 0.0])

    results = index.search(
        [0.0, 0.0],
        1,
        10
    )

    result_id, distance = results[0]

    assert isinstance(result_id, int)
    assert isinstance(distance, float)

def test_invalid_dimension():
    index = hnsw_cpp.HNSWIndex(3, 16, 64)

    with pytest.raises(ValueError):
        index.add([1.0, 2.0])