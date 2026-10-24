"""Original NumPy predicate parity and whole-frame boundary fallbacks."""
import numpy as np
import pytest
from eventwm_graph.l2_two_stage.view import certified_camera_frustum


def reference(xyz,origins,rotations):
    with np.errstate(all='ignore'):
        local=np.matmul(xyz[None]-origins[:,None],rotations);forward=-local[...,2]
        return (forward>1e-6)&(np.abs(local[...,0])<=forward)&(np.abs(local[...,1])<=forward)&(np.linalg.norm(local,axis=-1)<=10.)


def result(xyz,origins,rotations):
    mask,fallback=certified_camera_frustum(xyz,origins,rotations)
    if fallback.any():mask[fallback]=reference(xyz,origins[fallback],rotations[fallback])
    return mask,fallback


def test_random_proper_cameras_certify_identical_numpy_predicates():
    rng=np.random.default_rng(101);rotations=[]
    for _ in range(120):
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        rotations.append(q)
    rotations=np.asarray(rotations);origins=rng.normal(size=(120,3))*7
    for scale in (.1,1.,10.,1e6,1e150):
        xyz=rng.normal(size=(257,3))*scale
        actual,fallback=result(xyz,origins,rotations)
        np.testing.assert_array_equal(actual,reference(xyz,origins,rotations))
        if scale<=10:assert not fallback.any()


def test_ulp_near_plane_fov_and_range_request_original_full_frame():
    points=[]
    for value in (1e-6,1.,10.):
        for nearby in (np.nextafter(value,0.),value,np.nextafter(value,np.inf)):
            points.append([0.,0.,-nearby])
            if value==1.:
                points.extend([[nearby,0.,-1.],[-nearby,0.,-1.],[0.,nearby,-1.],[0.,-nearby,-1.]])
    xyz=np.asarray(points);origins=np.zeros((1,3));rotations=np.eye(3)[None]
    actual,fallback=result(xyz,origins,rotations)
    assert fallback.all()
    np.testing.assert_array_equal(actual,reference(xyz,origins,rotations))


def test_nonfinite_or_overflow_uses_the_original_frame_computation():
    xyz=np.array([[np.inf,0.,0.],[1e308,1e308,-1e308]])
    origins=np.zeros((2,3));rotations=np.repeat(np.eye(3)[None],2,axis=0)
    actual,fallback=result(xyz,origins,rotations)
    assert fallback.all()
    np.testing.assert_array_equal(actual,reference(xyz,origins,rotations))


def test_helper_rejects_dtypes_and_shapes_outside_its_float64_proof():
    valid=[np.zeros((2,3)),np.zeros((1,3)),np.eye(3)[None]]
    for i in range(3):
        arguments=list(valid);arguments[i]=arguments[i].astype(np.float32)
        with pytest.raises(ValueError,match='float64'):certified_camera_frustum(*arguments)
    for arguments in ((np.zeros((2,2)),valid[1],valid[2]),
                      (valid[0],np.zeros((2,3)),valid[2]),
                      (valid[0],valid[1],np.zeros((1,2,3)))):
        with pytest.raises(ValueError,match='requires XYZ'):certified_camera_frustum(*arguments)
