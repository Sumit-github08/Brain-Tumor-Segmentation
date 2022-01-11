import numpy as np
import streamlit as st
import pickle
import SimpleITK as sitk
import CUSTOM
from CUSTOM import Vnet_3d, SurvPredNet
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
import urllib

pretrained_model = "./Dataset/vnet_model_2.h5"
survival_model_file = "./Dataset/survival_model_files"
survival_model_weights = "./Dataset/survival_model.h5"
@st.cache
def model_out(input_img, input_msk, age):
    vnet_model = None
    vnet_model = Vnet_3d(128,128,128,3, 8,0.2)

    vnet_model.load_weights(pretrained_model)
    pred_image = vnet_model.predict(np.reshape(input_img, (1,128,128,128,3)))

    # Loading Final Survival_Prediction Model and predicting survival_days given age of patient
    survival_model_files = open(survival_model_file, 'rb')
    age_normalizer, survival_normalizer = pickle.load(survival_model_files)
    input_img_ = Input((128,128,128,3))
    age_m = Input((1))

    survival_model = SurvPredNet(input_img_,age_m)
    survival_model.load_weights(survival_model_weights)
    age = age_normalizer.transform(np.array(age).reshape(1,-1))

    pred_age = survival_model.predict((np.reshape(input_img, (1,128,128,128,3)), age))
    pred_age = survival_normalizer.inverse_transform(np.array(pred_age).reshape(-1,1))[0]
    return input_img, input_msk, pred_image, pred_age


def main():
    st.set_page_config(page_title='MRI Segmentation', page_icon="🧠")
    st.title("Brain Tumor Segmentation")
    st.image("https://www.med.upenn.edu/cbica/assets/user-content/images/BraTS/BRATS_banner_noCaption.png")
    
    
    # Sample Data
    st.write("Upload data:\n")
    input_img  = st.file_uploader("Upload Image", type ='npy')
    input_msk = st.file_uploader("Upload Mask", type ='npy')
    
    link = 'https://drive.google.com/drive/folders/1UO7cicW8qe2u2iqsdfJRX7POMVmZAIAs?usp=sharing'
    st.write("Download Sample Data "+ link)
    
    
    # SideBar
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app"])
    
    if app_mode == "Show instructions":
        if (input_img is not None) and (input_msk is not None):
            st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Run the app":
        if input_img is not None:
            input_img = np.load(input_img)
        if input_msk is not None:
            input_msk = np.load(input_msk)
        age = st.sidebar.select_slider("Patient Age ", np.arange(30,75, 1), 45)
        slice = st.sidebar.select_slider("Slice",np.arange(1,128,1),64)
        
        img, msk, pred_img, pred_age = model_out(input_img, input_msk, age)
        
        input_images = [input_img[:,:,slice,0], input_img[:,:,slice,1], input_img[:,:,slice,2]]
        
        st.write("## MRI input images of different modalities")
        fig, ax = plt.subplots(1,3, figsize = (10,30))
        ax[0].imshow(input_images[0])
        ax[0].set_title('Image t1ce')
        ax[1].imshow(input_images[1])
        ax[1].set_title('Image t2')
        ax[2].imshow(input_images[2])
        ax[2].set_title('Image flair')
        st.pyplot(fig)

        fig_3, ax = plt.subplots(1,2, figsize = (7,10))
        msk = np.argmax(msk, axis=-1)
        ax[0].imshow(msk[:,:,slice])
        ax[0].set_title('Mask Image')
        pred_img = np.argmax(pred_img[0], axis=-1)
        ax[1].imshow(pred_img[:,:,slice])
        ax[1].set_title('Predicted Mask image')
        st.pyplot(fig_3)
        st.write("## Predicted Survival Days of the Patient is {} days.".format(int(pred_age[0])))
        st.caption("Created by: \n Sumit Gulati")
        
if __name__=='__main__':
    main()
