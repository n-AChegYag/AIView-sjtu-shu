# Time : 2020.8
# Author : Chenwei Cai
# File : reg_sitk.py
#
# Intro : This is a image registration code for MICCAI MOOD 2020 based on SimpleITK. Two image types (brain and
#         abdomen) are applied to. The code is used for registering images before region detection and transforming
#         labels back to original space after region detection. The code supports affine registration and relative
#         pre-processing.
#
# Note : 1. The function registration_initialization() will prepare everything needed for brain and abdom registration,
#           so it should be executed only once.
#        2. fix_img_brain, fix_img_abdom and fix_seg_abdom are treated as global variable, which are loaded before
#           registration start.
#        3. The only four differences between brain and abdom registration are image_preprocessing(), inv_transform(),
#           brain_registration_execute() and abdom_registration_execute(). Changing the input variable data_string for
#           the first two functions is fine.
#        4. If the intensity range of label is {0, 1}, then choose sitk.sitkNearestNeighbor in inv_transform().
#           If the intensity range of label is [0, 1], then choose sitk.sitkLinear in inv_transform().
#
#
# Demo 1 (brain):
# registration_initialization()
# mov = sitk.ReadImage(".../00001.nii.gz", sitk.sitkFloat32)
# mov_info = get_image_information(mov)
# mov = image_preprocessing(mov, 'brain')
# aff_transform = brain_registration_execute(mov)
# mov_aff = fwd_transform(mov, aff_transform, sitk.sitkLinear)   # Map to the space of the fix image.
#
# #...Input mov_aff...
# #...Detection...
# #...Output mov_label
#
# mov_label_inv = inv_transform(mov_label, aff_transform, sitk.sitkNearestNeighbor, 'brain', mov_info) # Map back to ...
# #...the space of the original moving image.
# sitk.WriteImage(mov_label_inv, ".../00001_label.nii.gz")
#
#
#
# Demo 2 (abdom):
# registration_initialization()
# mov = sitk.ReadImage(".../00001.nii.gz", sitk.sitkFloat32)
# mov_info = get_image_information(mov)
# mov = image_preprocessing(mov, 'abdom')
# aff_transform = abdom_registration_execute(mov)
# mov_aff = fwd_transform(mov, aff_transform, sitk.sitkLinear)
#
# #...Input mov_aff...
# #...Detection...
# #...Output mov_label
#
# mov_label_inv = inv_transform(mov_label, aff_transform, sitk.sitkNearestNeighbor, 'abdom', mov_info)
# sitk.WriteImage(mov_label_inv, ".../00001_label.nii.gz")



import SimpleITK as sitk
# import pickle


def fix_data_initialization():
    global fix_img_brain, fix_img_abdom, fix_seg_abdom, brain_matrix_mask
    fix_img_brain = sitk.ReadImage("/workspace/00000brain.nii.gz", sitk.sitkFloat32)
    fix_img_abdom = sitk.ReadImage("/workspace/00008abdom.nii.gz", sitk.sitkFloat32)
    fix_seg_abdom = sitk.ReadImage("/workspace/00008abdseg.nii.gz", sitk.sitkFloat32)
    brain_matrix_mask = sitk.ReadImage("/workspace/brain_metric_mask.nii.gz", sitk.sitkUInt8)
    # code for pickle package:
    # fix_img_brain = sitk.Image()
    # fix_img_abdom = sitk.Image()
    # fix_seg_abdom = sitk.Image()
    # file = open('.../fix_img_brain.data', 'rb')
    # fix_img_brain = sitk.GetImageFromArray(pickle.load(file))
    # file = open('.../fix_img_abdom.data', 'rb')
    # fix_img_abdom = sitk.GetImageFromArray(pickle.load(file))
    # file = open('.../fix_seg_abdom', 'rb')
    # fix_seg_abdom = sitk.GetImageFromArray(pickle.load(file))
    # file.close()
    # fix_img_brain = image_preprocessing(fix_img_brain, 'brain')
    # fix_img_abdom = image_preprocessing(fix_img_abdom, 'abdom')
    # fix_seg_abdom = image_preprocessing(fix_seg_abdom, 'abdom')


def abdom_pre_transform_initialization():
    global pre_aff
    pre_aff = sitk.AffineTransform(3)
    pre_aff.SetCenter((0, 0, 0))
    pre_aff.SetMatrix([1.1, 0, 0, 0, 1.1, 0, 0, 0, 1.1])
    pre_aff.SetTranslation((19, 19, -21))

def brain_registration_initialization():
    global reg_method_brain
    reg_method_brain = sitk.ImageRegistrationMethod()
    # similarity
    reg_method_brain.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg_method_brain.SetMetricSamplingStrategy(reg_method_brain.RANDOM)
    reg_method_brain.SetMetricFixedMask(brain_matrix_mask)
    reg_method_brain.SetMetricMovingMask(brain_matrix_mask)
    reg_method_brain.SetMetricSamplingPercentage(percentage=0.5, seed=6666)  # SetMetricSamplingPercentagePerLevel
    # optimizer
    reg_method_brain.SetOptimizerAsConjugateGradientLineSearch(
        learningRate=0.1, numberOfIterations=500, convergenceMinimumValue=1e-6, convergenceWindowSize=5,
        lineSearchLowerLimit=0, lineSearchUpperLimit=2, lineSearchEpsilon=0.1, lineSearchMaximumIterations=50,
        estimateLearningRate=reg_method_brain.Once, maximumStepSizeInPhysicalUnits=0.1)
    reg_method_brain.SetOptimizerScalesFromPhysicalShift()
    # interpolator
    reg_method_brain.SetInterpolator(Interpolator=sitk.sitkLinear)
    # multi-level
    reg_method_brain.SetShrinkFactorsPerLevel(shrinkFactors=[5, 4])
    reg_method_brain.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    reg_method_brain.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg_method_brain.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                                lambda: command_multi_iteration(reg_method_brain))


def abdom_registration_initialization():
    global reg_method_abdom
    reg_method_abdom = sitk.ImageRegistrationMethod()
    # similarity
    reg_method_abdom.SetMetricAsMeanSquares()
    reg_method_abdom.SetMetricSamplingStrategy(reg_method_abdom.RANDOM)
    reg_method_abdom.SetMetricSamplingPercentage(percentage=0.5, seed=12321)  # SetMetricSamplingPercentagePerLevel
    # optimizer
    reg_method_abdom.SetOptimizerAsConjugateGradientLineSearch(
        learningRate=0.1, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=5,
        lineSearchLowerLimit=0, lineSearchUpperLimit=2, lineSearchEpsilon=0.1, lineSearchMaximumIterations=50,
        estimateLearningRate=reg_method_abdom.Once, maximumStepSizeInPhysicalUnits=0.1)
    reg_method_abdom.SetOptimizerScalesFromPhysicalShift()
    # interpolator
    reg_method_abdom.SetInterpolator(Interpolator=sitk.sitkLinear)
    # multi-level
    reg_method_abdom.SetShrinkFactorsPerLevel(shrinkFactors=[6, 4])
    reg_method_abdom.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    reg_method_abdom.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg_method_abdom.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                                lambda: command_multi_iteration(reg_method_abdom))


def abdom_skeleton_extract_initialization():
    global threshold1, threshold2
    threshold1 = sitk.ThresholdImageFilter()
    threshold1.SetOutsideValue(0)
    threshold1.SetLower(0.41)
    threshold2 = sitk.ThresholdImageFilter()
    threshold2.SetOutsideValue(1)
    threshold2.SetUpper(0.40)


def registration_initialization():
    print("Registration initialization start!")
    fix_data_initialization()
    brain_registration_initialization()
    abdom_pre_transform_initialization()
    abdom_skeleton_extract_initialization()
    abdom_registration_initialization()
    print("Registration initialization done!\n")


def abdomen_skeleton_extract(img):
    global threshold1, threshold2
    img_seg = threshold1.Execute(img)
    return threshold2.Execute(img_seg)


def brain_registration_execute(mov_img):
    global reg_method_brain, fix_img_brain
    if mov_img.GetSize() != (256, 256, 256):
        raise Exception("Input is not the brain data!")
    # transform setting
    initial_transform = sitk.CenteredTransformInitializer(
        fix_img_brain, mov_img, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg_method_brain.SetInitialTransform(initial_transform, inPlace=False)  # False
    print("Affine registration for brain data start!")
    affine_trans = reg_method_brain.Execute(fix_img_brain, mov_img)
    termination_reminder(reg_method_brain)
    return affine_trans


def abdom_registration_execute(mov_img):
    global reg_method_abdom, fix_img_abdom, fix_seg_abdom
    if mov_img.GetSize() != (512, 512, 512):
        raise Exception("Input is not the abdom data!")
    mov_seg = abdomen_skeleton_extract(mov_img)
    # transform setting for skeleton
    initial_transform = sitk.CenteredTransformInitializer(
        fix_seg_abdom, mov_seg, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg_method_abdom.SetInitialTransform(initial_transform, inPlace=False)  # False
    print("Affine registration for abdom skeleton start!")
    affine_trans = reg_method_abdom.Execute(fix_seg_abdom, mov_seg)
    termination_reminder(reg_method_abdom)
    mov_fwd1 = sitk.Resample(mov_img, affine_trans, sitk.sitkLinear)
    # transform setting for image
    initial_transform = sitk.CenteredTransformInitializer(
        fix_img_abdom, mov_fwd1, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg_method_abdom.SetInitialTransform(initial_transform, inPlace=False)  # False
    print("Affine registration for abdom data start!")
    affine_trans1 = reg_method_abdom.Execute(fix_img_abdom, mov_fwd1)
    termination_reminder(reg_method_abdom)
    affine_trans.AddTransform(affine_trans1)
    return affine_trans


def fwd_transform(mov_img, affine_trans, interpolator):
    return sitk.Resample(mov_img, affine_trans, interpolator)


def inv_transform(mov_fwd, affine_trans, interpolator, data_string, img_info):
    # If mov_fwd is a segmentation, you should choose 'NearestNeighbor'.
    # If mov_fwd is a image, you are recommended to choose 'Linear'.
    mov_fwd.SetOrigin(img_info[0])
    mov_fwd.SetSpacing(img_info[1])
    mov_fwd.SetDirection(img_info[2])
    if data_string == "brain":
        mov_inv = sitk.Resample(mov_fwd, affine_trans.GetInverse(), interpolator)
    elif data_string == "abdom":
        global pre_aff
        post_aff = sitk.Transform()
        post_aff.AddTransform(pre_aff)
        post_aff.AddTransform(affine_trans)
        mov_inv = sitk.Resample(mov_fwd, post_aff.GetInverse(), interpolator)
    else:
        raise Exception("We don't have a data type called {}.".format(data_string))
    return mov_inv


def get_image_information(img):
    return [img.GetOrigin(), img.GetSpacing(), img.GetDirection()]


def image_preprocessing(img, data_string):
    img.SetOrigin((0, 0, 0))
    img.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    if data_string == "brain":
        img.SetSpacing((0.7, 0.7875, 0.7))
    elif data_string == "abdom":
        img.SetSpacing((0.75, 0.75, 0.84))
        global pre_aff
        img = sitk.Resample(img, pre_aff, sitk.sitkNearestNeighbor)
    else:
        raise Exception("We don't have a data type called {}.".format(data_string))
    return img


def command_multi_iteration(reg_method):
    if reg_method.GetCurrentLevel() > 0:
        print("Optimizer stop condition: {0}".format(reg_method.GetOptimizerStopConditionDescription()))
        print("Iteration: {0}".format(reg_method.GetOptimizerIteration()))
        print("Metric value: {0}".format(reg_method.GetMetricValue()))
        print("--------- Resolution Changing ---------")


def termination_reminder(reg_method):
    print("Optimizer stop condition: {0}".format(reg_method.GetOptimizerStopConditionDescription()))
    print("Iteration: {0}".format(reg_method.GetOptimizerIteration()))
    print("Metric value: {0}".format(reg_method.GetMetricValue()))
    print("--------- Done ---------\n")


