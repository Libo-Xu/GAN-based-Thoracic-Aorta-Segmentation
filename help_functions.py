#!/usr/bin/env python
# coding: utf-8

# In[12]:


import SimpleITK as sitk
import numpy as np
import os
import math
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology
from scipy.ndimage.interpolation import zoom


def upsample(array, if_image=False):
    chs = array.shape[0]
    h = array.shape[1]
    w = array.shape[2]
    array = np.reshape(array, (chs,h,w))
    tmp = np.zeros((chs,h*2,w*2))
    if if_image:
        order_ = 1
    else:
        order_ = 0
    for i in range(chs):
        tmp[i] = zoom(array[i], zoom = 2, order=order_)

    return tmp

def binary(array):
    array[array>=0.5] = 1
    array[array<0.5] = 0
    return array


# 下面三个是提取边界和计算最小距离的实用函数
def get_surface(mask, voxel_spacing):
    """
    :param mask: ndarray
    :param voxel_spacing: 体数据的spacing
    :return: 提取array的表面点的真实坐标(以mm为单位)
    """

    # 卷积核采用的是三维18邻域

    kernel = morphology.generate_binary_structure(3, 2)
    surface = morphology.binary_erosion(mask, kernel) ** mask

    surface_pts = surface.nonzero()

    surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))

    # (0.7808688879013062, 0.7808688879013062, 2.5) (88, 410, 512)
    # 读出来的数据spacing和shape不是对应的,所以需要反向
    return surface_pts * np.array(voxel_spacing[::-1]).reshape(1, 3)

def get_pred2real_nn(real_mask_surface_pts, pred_mask_surface_pts):
    """
    :return: 预测结果表面体素到金标准表面体素的最小距离
    """

    tree = spatial.cKDTree(real_mask_surface_pts)
    nn, _ = tree.query(pred_mask_surface_pts)

    return nn

def get_real2pred_nn(real_mask_surface_pts, pred_mask_surface_pts):
    """
    :return: 金标准表面体素到预测结果表面体素的最小距离
    """
    tree = spatial.cKDTree(pred_mask_surface_pts)
    nn, _ = tree.query(real_mask_surface_pts)

    return nn

def get_ASSD(real_mask, pred_mask, voxel_spacing):
    """
    :return: 对称位置平均表面距离 Average Symmetric Surface Distance
    """
    real_mask_surface_pts = get_surface(real_mask, voxel_spacing)
    pred_mask_surface_pts = get_surface(pred_mask, voxel_spacing)
    pred2real_nn = get_pred2real_nn(real_mask_surface_pts, pred_mask_surface_pts)
    real2pred_nn = get_real2pred_nn(real_mask_surface_pts, pred_mask_surface_pts)
    return (pred2real_nn.sum() + real2pred_nn.sum()) / \
           (real_mask_surface_pts.shape[0] + pred_mask_surface_pts.shape[0])

def saveNumpyImageToITKImage(templatefile, outputfile, ndarray):
    template_image = sitk.ReadImage(templatefile)
    itk_image = sitk.GetImageFromArray(ndarray)
    itk_image = reorientITKImageFromRAI(itk_image, template_image)
    outsize = template_image.GetSize()
    itk_image = resampleITKImageLinear(itk_image, outsize)
    itk_image.CopyInformation(template_image)
    sitk.WriteImage(itk_image, outputfile)
    return itk_image

def resampleITKImage(itk_image, output_size, interpolation):
    sz1, sp1, origin = itk_image.GetSize(), itk_image.GetSpacing(), itk_image.GetOrigin()
    direction = itk_image.GetDirection()
    num_dim = len(sz1)
    newsize = [0]*num_dim
    for dim in range(num_dim):
        if output_size[dim] < 0:
            newsize[dim] = int(round((-sz1[dim] * sp1[dim]) / output_size[dim]))
        elif output_size[dim] == 0:
            newsize[dim] = sz1[dim]
        else:
            newsize[dim] = output_size[dim]
    newscale = [float(b) / float(a) for a, b in zip(sz1, newsize)]
    newspacing = [a / b for a, b in zip(sp1, newscale)]
    t = sitk.Transform(num_dim, sitk.sitkScale)
    scaling_factors = [1.]*num_dim
    t.SetParameters(scaling_factors)
    itk_image = sitk.Resample(itk_image, newsize, t, interpolation,
                              origin, newspacing, direction, 0.0, sitk.sitkFloat32)

    return itk_image

def getAxisOrderToRAI( direction ):
    axis = np.identity(3)
    border_direction = np.array(direction)
    border_direction = np.reshape(border_direction,[3,3])
    axis_order = [0]*3
    # print(border_direction)
    for b_i in range(3):
        projects = np.zeros(3)
        vec = border_direction[b_i]
        for dim in range(3):
            projects[dim] = np.dot(axis[dim], vec)
        projects = np.abs(projects)
        axis_order[b_i] = int(np.argmax(projects))
    return axis_order

def getFlipMap(direction):
    flip_map = [False, False, False]
    axis = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    for dim in range(3):
        projection = np.dot(axis[dim], direction[dim * 3:(dim + 1) * 3])
        if projection < 0:
            flip_map[dim] = True
    # print(flip_map)
    return flip_map
 
def reorientITKImageFromRAI(itk_image, template_image):
    direction = template_image.GetDirection()
    axis_order = getAxisOrderToRAI(direction)
    temporary_image = sitk.PermuteAxes(template_image, axis_order)
    direction = temporary_image.GetDirection()
    flip_map = getFlipMap(direction)

    itk_image = sitk.Flip(itk_image, flip_map)
    reversed_axis_order = [0] * 3
    for dim in range(3):
        reversed_axis_order[axis_order[dim]] = dim
    itk_image = sitk.PermuteAxes(itk_image, reversed_axis_order)
    return itk_image
 
def resampleITKImageLinear(itk_image, output_size):
    interploation = sitk.sitkLinear;
    itk_image = resampleITKImage(itk_image, output_size, interploation)
    return itk_image

