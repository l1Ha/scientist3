import torch

from cartan_dec.masking import geometry_masking, make_mask, random_masking, span_masking


def test_random_masking_respects_valid_and_keeps_context():
    valid = torch.ones(4, 50, dtype=torch.bool)
    valid[0, 40:] = False
    masked = random_masking(valid, ratio=0.5)
    assert not (masked & ~valid).any()
    assert (masked.sum(dim=1) > 0).all()
    assert ((valid & ~masked).sum(dim=1) >= 1).all()
    assert (masked.sum(dim=1).float() / valid.sum(dim=1).float() - 0.5).abs().max() < 0.1


def test_span_masking_produces_contiguous_spans():
    valid = torch.ones(2, 60, dtype=torch.bool)
    masked = span_masking(valid, ratio=0.4, span_length=10)
    assert not (masked & ~valid).any()
    assert (masked.sum(dim=1) > 0).all()
    # At least one run of length > 1 should exist.
    runs = (masked[:, 1:] & masked[:, :-1]).any()
    assert bool(runs)


def test_geometry_masking_prefers_high_scores():
    valid = torch.ones(1, 30, dtype=torch.bool)
    defect = torch.zeros(1, 30)
    defect[0, 15] = 100.0
    masked = geometry_masking(valid, defect, ratio=0.2, span_length=5, hard=True)
    assert masked[0, 15]
    assert masked.sum() > 0


def test_make_mask_dispatch():
    valid = torch.ones(1, 20, dtype=torch.bool)
    for strategy in ("random", "span"):
        masked = make_mask(valid, strategy=strategy, ratio=0.3, span_length=4)
        assert masked.shape == valid.shape
        assert masked.any()
