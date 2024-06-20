import nibabel
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
from nilearn.input_data import MultiNiftiMasker
from nilearn_sandbox._utils.map_alignment import align_many_to_one_nii, \
    align_list_with_last_nii, _align_one_to_one_flat, _align_many_to_one_flat


def test_align_many_to_one_nii():
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    a = rng.randn(10, 5 * 5 * 5)
    b = rng.permutation(a)
    c = rng.permutation(a)
    masker = MultiNiftiMasker(mask_img=nibabel.Nifti1Image(np.ones((5, 5, 5)),
                                                           affine=affine))
    masker.fit()
    img_a = masker.inverse_transform(a)
    img_b = masker.inverse_transform(b)
    img_c = masker.inverse_transform(c)
    new_img_b = align_many_to_one_nii(masker, img_a, img_b)
    new_b = masker.transform(new_img_b)
    assert_array_almost_equal(a, new_b)
    results = align_list_with_last_nii(masker, (img_b, img_c, img_a))
    new_b = masker.transform(results[0])
    new_c = masker.transform(results[1])
    assert_array_almost_equal(a, new_b)
    assert_array_almost_equal(a, new_c)


def test_align_one_to_one_flat():
    rng = np.random.RandomState(0)
    a = rng.rand(10, 100)
    a_copy = a.copy()
    b = rng.permutation(a)
    b_copy = b.copy()
    c = _align_one_to_one_flat(a, b, inplace=False)
    assert_array_almost_equal(a, c)
    assert_array_almost_equal(a, a_copy)
    assert_array_almost_equal(b, b_copy)
    _align_one_to_one_flat(a, b, inplace=True)
    assert_array_almost_equal(a, b)
    assert_array_almost_equal(a, a_copy)
    assert_raises(AssertionError, assert_array_almost_equal, b, b_copy)


def test_align_flat():
    rng = np.random.RandomState(0)
    ref = rng.rand(10, 100)
    b = rng.permutation(ref)
    c = rng.permutation(ref)
    target_list = [b, c]
    target_list_copy = [b.copy(), c.copy()]
    aligned_target_list = _align_many_to_one_flat(ref, target_list,
                                                  inplace=False)
    for target, target_copy in zip(target_list, target_list_copy):
        assert_array_almost_equal(target, target_copy)
    for target in aligned_target_list:
        assert_array_almost_equal(ref, target)