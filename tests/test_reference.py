"""# AI-assisted (ChatGPT): starter tests; review and extend."""
import torch
from validation.reference import reference_attention, compare_outputs


def test_identical_values():
    q=torch.randn(1,3,4)
    k=torch.randn(1,3,4)
    v=torch.ones(1,3,2)
    out=reference_attention(q,k,v)
    assert torch.allclose(out,torch.ones_like(out),atol=1e-12)


def test_causal_first_token_uses_only_first_value():
    q=torch.randn(1,3,4)
    k=torch.randn(1,3,4)
    v=torch.tensor([[[3.,5.],[7.,9.],[11.,13.]]])
    out=reference_attention(q,k,v,causal=True)
    assert torch.allclose(out[0,0],v[0,0].double(),atol=1e-12)


def test_numerical_error_zero_for_identical_arrays():
    x=torch.randn(2,3)
    result=compare_outputs(x,x)
    assert result['max_absolute_error']==0


def test_causal_single_token():
    q = torch.tensor([[[1., 0.]]])
    k = q.clone()
    v = torch.tensor([[[7., 8.]]])
    assert torch.allclose(reference_attention(q, k, v, causal=True), v.double())


def test_batch_and_heads_supported():
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 4, 8)
    v = torch.randn(2, 3, 4, 6)
    assert reference_attention(q, k, v).shape == (2, 3, 4, 6)


def test_reference_matches_pytorch_small_fp64():
    import torch.nn.functional as F
    q = torch.randn(2, 5, 8, dtype=torch.float64)
    k = torch.randn(2, 5, 8, dtype=torch.float64)
    v = torch.randn(2, 5, 4, dtype=torch.float64)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(reference_attention(q, k, v, causal=True), expected, atol=1e-12, rtol=1e-12)


def test_mismatched_dimensions_rejected():
    import pytest
    with pytest.raises(ValueError):
        reference_attention(torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 3))
