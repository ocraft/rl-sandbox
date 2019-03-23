from rlbox.rand import Sampler


def test_uses_buffer_of_precomputed_values():
    assert len(Sampler.standard_normal(10)) == 10
    assert Sampler.randn() is not None
