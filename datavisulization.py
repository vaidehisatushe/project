import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
import os

def create_data_visulization():

    train_data_path = "D:/Shivank/BraTS_Brain_Segmentation/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/"
    valid_data_path = "D:/Shivank/BraTS_Brain_Segmentation/MICCAI2024-BraTS-GoAT-ValidationData/MICCAI2024-BraTS-GoAT-ValidationData/"

    sample_filename = 'D:/Shivank/BraTS_Brain_Segmentation/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/BraTS-GoAT-00000/BraTS-GoAT-00000-t2w.nii.gz'
    sample_filename_mask = 'D:/Shivank/BraTS_Brain_Segmentation/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/BraTS-GoAT-00000/BraTS-GoAT-00000-seg.nii.gz'

    sample_img = nib.load(sample_filename)
    sample_img = np.asanyarray(sample_img.dataobj)
    sample_img = np.rot90(sample_img)
    sample_mask = nib.load(sample_filename_mask)
    sample_mask = np.asanyarray(sample_mask.dataobj)
    sample_mask = np.rot90(sample_mask)
    print("img shape ->", sample_img.shape)
    print("mask shape ->", sample_mask.shape)

    case_id = 'BraTS-GoAT-00000'

    # Loading MRI images and segmentation mask
    test_image_flair = nib.load(os.path.join(train_data_path, case_id, case_id + '-t2w.nii.gz')).get_fdata()
    test_image_t1    = nib.load(os.path.join(train_data_path, case_id, case_id + '-t1n.nii.gz')).get_fdata()
    test_image_t1ce  = nib.load(os.path.join(train_data_path, case_id, case_id + '-t1c.nii.gz')).get_fdata()
    test_image_t2    = nib.load(os.path.join(train_data_path, case_id, case_id + '-t2f.nii.gz')).get_fdata()
    test_mask        = nib.load(os.path.join(train_data_path, case_id, case_id + '-seg.nii.gz')).get_fdata()

    # Check dimensions of loaded data
    print("FLAIR image shape:", test_image_flair.shape)
    print("T1 image shape   :", test_image_t1.shape)
    print("T1CE image shape :", test_image_t1ce.shape)
    print("T2 image shape   :", test_image_t2.shape)
    print("Mask image shape :", test_mask.shape)

    # Value counts of labels in the mask
    label_values, label_counts = np.unique(test_mask, return_counts = True)
    label_counts_dict = dict(zip(label_values, label_counts))
    print("\nLabel Value Counts:")
    print(label_counts_dict)

    # Visualizing MRI images and segmentation mask
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (18, 12))
    slice_idx = test_image_flair.shape[2] // 2

    # FLAIR
    ax1.imshow(test_image_flair[:, :, slice_idx], cmap = 'gray')
    ax1.set_title('Image FLAIR')

    # T1
    ax2.imshow(test_image_t1[:, :, slice_idx], cmap = 'gray')
    ax2.set_title('Image T1')

    # T1CE
    ax3.imshow(test_image_t1ce[:, :, slice_idx], cmap = 'gray')
    ax3.set_title('Image T1CE')

    # T2
    ax4.imshow(test_image_t2[:, :, slice_idx], cmap = 'gray')
    ax4.set_title('Image T2')

    # MASK
    ax5.imshow(test_mask[:, :, slice_idx], cmap = 'gray')
    ax5.set_title('Mask')

    plt.show()


    # In[6]:


    case_id = 'BraTS-GoAT-00000'

    # Loading MRI images and segmentation mask
    test_image_flair = nib.load(os.path.join(train_data_path, case_id, case_id + '-t2w.nii.gz')).get_fdata()
    test_mask        = nib.load(os.path.join(train_data_path, case_id, case_id + '-seg.nii.gz')).get_fdata()

    # Visualizing MRI images and segmentation mask
    fig, axes = plt.subplots(1, 2, figsize = (15, 7))

    # FLAIR image
    axes[0].imshow(test_image_flair[:, :, slice_idx], cmap = 'gray')
    axes[0].set_title('FLAIR Image')

    # Overlay mask on FLAIR image
    overlay_mask = np.ma.masked_where(test_mask == 0, test_mask)
    axes[1].imshow(test_image_flair[:, :, slice_idx], cmap = 'gray')
    axes[1].imshow(overlay_mask[:, :, slice_idx], cmap = 'cool', alpha = 0.5)
    axes[1].set_title('FLAIR Image with Overlay Mask')

    plt.show()

    case_id = 'BraTS-GoAT-00003'

    # Loading MRI images and segmentation mask
    test_image_flair = nib.load(os.path.join(train_data_path, case_id, case_id + '-t2w.nii.gz')).get_fdata()
    test_image_t1    = nib.load(os.path.join(train_data_path, case_id, case_id + '-t1n.nii.gz')).get_fdata()
    test_image_t1ce  = nib.load(os.path.join(train_data_path, case_id, case_id + '-t1c.nii.gz')).get_fdata()
    test_image_t2    = nib.load(os.path.join(train_data_path, case_id, case_id + '-t2f.nii.gz')).get_fdata()
    test_mask        = nib.load(os.path.join(train_data_path, case_id, case_id + '-seg.nii.gz')).get_fdata()

    # Check dimensions of loaded data
    print("FLAIR image shape:", test_image_flair.shape)
    print("T1 image shape   :", test_image_t1.shape)
    print("T1CE image shape :", test_image_t1ce.shape)
    print("T2 image shape   :", test_image_t2.shape)
    print("Mask image shape :", test_mask.shape)

    # Value counts of labels in the mask
    label_values, label_counts = np.unique(test_mask, return_counts = True)
    label_counts_dict = dict(zip(label_values, label_counts))
    print("\nLabel Value Counts:")
    print(label_counts_dict)

    # Visualizing MRI images and segmentation mask
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (18, 12))
    slice_idx = test_image_flair.shape[2] // 2

    # FLAIR
    ax1.imshow(test_image_flair[:, :, slice_idx], cmap = 'gray')
    ax1.set_title('Image FLAIR')

    # T1
    ax2.imshow(test_image_t1[:, :, slice_idx], cmap = 'gray')
    ax2.set_title('Image T1')

    # T1CE
    ax3.imshow(test_image_t1ce[:, :, slice_idx], cmap = 'gray')
    ax3.set_title('Image T1CE')

    # T2
    ax4.imshow(test_image_t2[:, :, slice_idx], cmap = 'gray')
    ax4.set_title('Image T2')

    # MASK
    ax5.imshow(test_mask[:, :, slice_idx], cmap = 'gray')
    ax5.set_title('Mask')

    plt.show()


    # In[8]:


    case_id = 'BraTS-GoAT-00007'

    # Loading MRI images and segmentation mask
    test_image_flair = nib.load(os.path.join(train_data_path, case_id, case_id + '-t2w.nii.gz')).get_fdata()
    test_image_t1    = nib.load(os.path.join(train_data_path, case_id, case_id + '-t1n.nii.gz')).get_fdata()
    test_image_t1ce  = nib.load(os.path.join(train_data_path, case_id, case_id + '-t1c.nii.gz')).get_fdata()
    test_image_t2    = nib.load(os.path.join(train_data_path, case_id, case_id + '-t2f.nii.gz')).get_fdata()
    test_mask        = nib.load(os.path.join(train_data_path, case_id, case_id + '-seg.nii.gz')).get_fdata()

    # Check dimensions of loaded data
    print("FLAIR image shape:", test_image_flair.shape)
    print("T1 image shape   :", test_image_t1.shape)
    print("T1CE image shape :", test_image_t1ce.shape)
    print("T2 image shape   :", test_image_t2.shape)
    print("Mask image shape :", test_mask.shape)

    # Value counts of labels in the mask
    label_values, label_counts = np.unique(test_mask, return_counts = True)
    label_counts_dict = dict(zip(label_values, label_counts))
    print("\nLabel Value Counts:")
    print(label_counts_dict)

    # Visualizing MRI images and segmentation mask
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (18, 12))
    slice_idx = test_image_flair.shape[2] // 2

    # FLAIR
    ax1.imshow(test_image_flair[:, :, slice_idx], cmap = 'gray')
    ax1.set_title('Image FLAIR')

    # T1
    ax2.imshow(test_image_t1[:, :, slice_idx], cmap = 'gray')
    ax2.set_title('Image T1')

    # T1CE
    ax3.imshow(test_image_t1ce[:, :, slice_idx], cmap = 'gray')
    ax3.set_title('Image T1CE')

    # T2
    ax4.imshow(test_image_t2[:, :, slice_idx], cmap = 'gray')
    ax4.set_title('Image T2')

    # MASK
    ax5.imshow(test_mask[:, :, slice_idx], cmap = 'gray')
    ax5.set_title('Mask')

    plt.show()
       
    # skip 50:-50 slices since there is not much to see
    fig, ax1 = plt.subplots(1, 1, figsize = (15, 15))
    ax1.imshow(rotate(montage(test_image_t1[50:-50,:,:]), 90, resize = True), cmap = 'gray')


    # In[10]:


    # skip 50:-50 slices since there is not much to see
    fig, ax1 = plt.subplots(1, 1, figsize = (15, 15))
    ax1.imshow(rotate(montage(test_mask[50:-50,:,:]), 90, resize = True), cmap = 'gray')


    # In[11]:


    print(test_image_flair.max())
    print(np.unique(test_mask))


    # ### Evaluation Matrix

    # In[12]:


    def setup_data_visulization():
        #visulization()
        create_data_visulization()

    if __name__ == '__main__':
        setup_data_visulization()