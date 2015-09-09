import nibabel
import numpy as np
from numpy.testing import assert_array_almost_equal, \
    assert_raises
from sklearn.utils.linear_assignment_ import linear_assignment
from nilearn.input_data import MultiNiftiMasker
from nilearn._utils import check_niimg_4d


def _spatial_correlation_flat(these_components, those_components):
    """Compute the spatial covariance betwwen two 2D ndarray

    Parameters
    ----------
    these_components: ndarray
        Shape (n_compoennts, n_features) First component set
    those_components: ndarray
        Shape (n_components, n_features) Second component set

    Returns
    -------
    corr: ndarray,
        Shape (n_components, n_components) : correlation matrix

    """
    this_S = np.sqrt(np.sum(these_components ** 2, axis=1))
    this_S[this_S == 0] = 1
    these_components /= this_S[:, np.newaxis]

    that_S = np.sqrt(np.sum(those_components ** 2, axis=1))
    that_S[that_S == 0] = 1
    those_components /= that_S[:, np.newaxis]
    corr = these_components.dot(those_components.T)
    these_components *= this_S[:, np.newaxis]
    those_components *= that_S[:, np.newaxis]
    return corr


def _align_many_to_one_flat(reference, target_list, inplace=False):
    """Align target_list with reference using linear_assignment"""
    if not isinstance(target_list, list):
        return _align_one_to_one_flat(reference, target_list, inplace=inplace)
    if not inplace:
        res = []
    for i, target_components in enumerate(target_list):
        if inplace:
            _align_one_to_one_flat(reference, target_components,
                                  inplace=True)
        else:
            res.append(_align_one_to_one_flat(reference, target_components,
                                             inplace=False))
    if inplace:
        res = target_list
    return res


def _align_one_to_one_flat(base_components, target_components, inplace=False):
    """Align target_components with base_components using linear_assignment"""
    indices = linear_assignment(-_spatial_correlation_flat(base_components,
                                                         target_components))

    if inplace:
        target_components[indices[:, 0]] = target_components[indices[:, 1]]
    else:
        new_target = np.empty_like(target_components)
        new_target[indices[:, 0]] = target_components[indices[:, 1]]
        target_components = new_target
    return target_components


def spatial_correlation(masker, this_img, that_img):
    """Compute the spatial covariance betwwen two 2D ndarray

    Parameters
    ----------
    these_components: nii-like image,
        4D image with n_components i the 4-th dim : First component set
    those_components: nii-like image,
        4D image with n_components i the 4-th dim : Second component set

    Returns
    -------
    corr: ndarray,
        Shape (n_components, n_components) : correlation matrix
    """
    this_flat = masker.transform(this_img)
    that_flat = masker.transform(that_img)
    return _spatial_correlation_flat(this_flat, that_flat)


def align_many_to_one_nii(masker, reference_img, target_imgs):
    """Align provided Nifti1Image with a reference, unmasking data using
    provided mask

    Parameters
    ----------
        masker: BaseMasker,
            Masker used to unmask provided Nifti1Image

        reference_img: nii-like image,
            Component map used as reference for alignment

        target_imgs: list of nii-like images,
            Components maps to be aligned

    Returns
    -------
        new_target_imgs: list of nii-like images,
            Aligned components maps, in the same order as provided
    """
    reference_flat = masker.transform(reference_img)
    target_flats = masker.transform(target_imgs)
    _align_many_to_one_flat(reference_flat, target_flats, inplace=True)
    if isinstance(target_flats, list):
        return [masker.inverse_transform(target_flat) for target_flat
                in target_flats]
    else:
        return masker.inverse_transform(target_flats)


def align_list_with_last_nii(masker, imgs):
    """Align provided Nifti1Image with last element of the list

    Parameters
    ----------
        masker: BaseMasker,
            Masker used to unmask provided Nifti1Image

        imgs: list of nii-like images,
            Components maps to be aligned with their last element

    Returns
    -------
        new_imgs: list of nii-like images,
            Aligned components maps, in the same order as provided
    """
    new_imgs = align_many_to_one_nii(masker, imgs[-1], imgs[:-1])
    new_imgs.append(check_niimg_4d(imgs[-1]))
    return new_imgs


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


# TODO test spatial correlation
