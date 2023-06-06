# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import pytest
import torch

import xformers.components.attention.attention_patterns as AP


# baseline implementations
def _local_1d_pattern(attn_size: int, window_size: int) -> torch.Tensor:
    assert (
        window_size % 2 == 1
    ), "The window size is assumed to be odd (counts self-attention + 2 wings)"
    h_win_size = window_size // 2

    attn_shape = (attn_size, attn_size)
    full_attn = torch.ones(attn_shape, dtype=torch.bool)

    mask = torch.tril(full_attn, diagonal=h_win_size)
    mask &= ~torch.tril(full_attn, diagonal=-(h_win_size + 1))
    return mask


def _generate_2d_grid(H, W):
    i = torch.arange(H)
    j = torch.arange(W)
    i, j = torch.meshgrid(i, j)
    return i, j


def _horizontal_axial_2d_distance(H, W, p=2.0):
    i, _ = _generate_2d_grid(H, W)
    ij = i.reshape(-1, 1).float()
    d = torch.cdist(ij, ij, p=p)
    return d


def _vertical_axial_2d_distance(H, W, p=2.0):
    _, j = _generate_2d_grid(H, W)
    ij = j.reshape(-1, 1).float()
    d = torch.cdist(ij, ij, p=p)
    return d


def _local_2d_distance(H, W, p=2.0):
    # axial is a special case with p=0 and distance=2
    i, j = _generate_2d_grid(H, W)
    ij = torch.stack([i.flatten(), j.flatten()], 1).float()
    d = torch.cdist(ij, ij, p=p)
    return d


def _local_2d_gaussian_distribution(H, W, sigma=1.0):
    d = _local_2d_distance(H, W, p=2.0) ** 2
    d = torch.exp(-0.5 * sigma ** (-2.0) * d)
    return d


@pytest.mark.parametrize("window_size", [3, 7, 11])
@pytest.mark.parametrize("attn_size", [50, 51, 64])
def test_local_1d_pattern(attn_size, window_size):
    mask = AP.local_1d_pattern(attn_size, window_size).float()
    mask_ref = _local_1d_pattern(attn_size, window_size).float()
    assert torch.allclose(mask, mask_ref)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_horizontal_axial_2d_distance(H, W, p):
    d = AP.horizontal_axial_2d_distance(H, W, p=p)
    d_ref = _horizontal_axial_2d_distance(H, W, p=p)
    assert torch.allclose(d, d_ref)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_vertical_axial_2d_distance(H, W, p):
    d = AP.vertical_axial_2d_distance(H, W, p=p)
    d_ref = _vertical_axial_2d_distance(H, W, p=p)
    assert torch.allclose(d, d_ref)


@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_local_2d_distance(H, W, p):
    d = AP.local_2d_distance(H, W, p=p)
    d_ref = _local_2d_distance(H, W, p=p)
    assert torch.allclose(d, d_ref)


@pytest.mark.parametrize("sigma", [0.5, 1, 2])
@pytest.mark.parametrize("W", [5, 7, 10])
@pytest.mark.parametrize("H", [5, 7, 10])
def test_local_2d_gaussian_distribution(H, W, sigma):
    d = AP.local_2d_gausian_distribution(H, W, sigma=sigma)
    d_ref = _local_2d_gaussian_distribution(H, W, sigma=sigma)
    assert torch.allclose(d, d_ref)


@pytest.mark.parametrize("window_size", [2, 4])
@pytest.mark.parametrize("W", [8, 16])
@pytest.mark.parametrize("H", [8, 16])
def test_swin_attention_pattern(H, W, window_size):
    # test non-shifted case
    d = AP.swin_attention_pattern(H, W, window_size, shift_size=0)

    # partition the self-attention into regions of window_size
    # similar to the window_partition function from the original paper
    h = H // window_size
    w = W // window_size
    d = d.reshape(h, window_size, w, window_size, h, window_size, w, window_size)

    for y, x in itertools.product(range(h), range(w)):
        # every region should fully attend to itself
        assert torch.all(d[y, :, x, :, y, :, x, :])
        for y2, x2 in itertools.product(range(h), range(w)):
            if y == y2 or x == x2:
                continue
            # different regions shouldn't attend between each other
            assert torch.all(~d[y, :, x, :, y2, :, x2, :])

    # test shifted case
    # in the shifted case, the self-attention should be the same
    # as in the non-shifted case, when we pad the inputs, apply the operations and then
    # remove the padding from the result
    d_shifted = AP.swin_attention_pattern(
        H, W, window_size, shift_size=window_size // 2
    )

    # add padding and remove shift
    h = H + window_size
    w = W + window_size
    d_padded = AP.swin_attention_pattern(h, w, window_size, shift_size=0)
    d_padded = d_padded.reshape(h, w, h, w)

    # remove padding elements
    half_size = window_size // 2
    s = slice(half_size, -half_size)
    d_padded = d_padded[s, s, s, s].reshape(H * W, H * W)

    assert torch.all(d_padded == d_shifted)


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("W", [8, 15])
@pytest.mark.parametrize("H", [8, 15])
def test_dilated_2d_pattern(H, W, k):
    d = AP.dilated_2d_pattern(H, W, k)
    d = d.reshape(H, W, H, W)
    for h, w in itertools.product(range(H), range(W)):
        i = h % k
        j = w % k
        # every kth element is taken
        assert torch.all(d[h, w][i::k, j::k])
        for ii, jj in itertools.product(range(k), range(k)):
            if ii == i and jj == j:
                continue
            # and the other elements are discarded
            assert torch.all(~d[h, w][ii::k, jj::k])


def test_pattern_to_layout():
    BLOCK = 16
    SIZE = 128
    LAYOUT_SIZE = SIZE // BLOCK

    # All ones
    mask1 = torch.ones((SIZE, SIZE), dtype=torch.bool)
    layout1 = AP.pattern_to_layout(mask1, BLOCK)
    ref1 = torch.ones((LAYOUT_SIZE, LAYOUT_SIZE), dtype=torch.long)
    assert torch.allclose(layout1, ref1)

    # Diagonal -> expect block diagonal
    mask2 = torch.eye(SIZE, dtype=torch.bool)
    layout2 = AP.pattern_to_layout(mask2, BLOCK)
    ref2 = torch.eye(LAYOUT_SIZE, dtype=torch.long)
    assert torch.allclose(layout2, ref2)

    # Lower triangular, without the diagonal
    # note that the layout will need to have the diagonal, else the coefficients close enough would not be computed
    mask3 = torch.tril(torch.ones((SIZE, SIZE)), diagonal=-1).to(torch.bool)
    layout3 = AP.pattern_to_layout(mask3, BLOCK)
    ref3 = torch.tril(torch.ones((LAYOUT_SIZE, LAYOUT_SIZE)), diagonal=0).to(torch.long)
    assert torch.allclose(layout3, ref3)

    # Handle heads properly
    mask = torch.cat((mask1, mask2, mask3))
    layout = AP.pattern_to_layout(mask, BLOCK)
    assert torch.allclose(layout, torch.cat((ref1, ref2, ref3)))

    # Catch problematic dimensions
    mask_off = torch.ones((SIZE + 3, SIZE), dtype=torch.bool)
    with pytest.raises(AssertionError):
        AP.pattern_to_layout(mask_off, BLOCK)


def test_alibi_pattern():
    mask = AP.alibi_pattern(1e-3, (16, 128, 128))
    # Minor, check that all the top left corners are True
    assert torch.sum(mask[:, 0, 0]) == 16
