import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
import math
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(r'C:\Users\DELL\streamlit-intel-image\mnet-best-model.hdf5')
  return model


with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Intel Image Classification
         """
         )

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png","jpeg"])

import cv2
from PIL import Image
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
col1, col2 = st.columns(2)

if file is None:
    st.text("Please upload an image file")
else:
    with col1:
      image = np.array(Image.open(file))
      IMG_SIZE = (224,224)
      img = cv2.resize(image,IMG_SIZE)
      img=img/255.0
      img = np.expand_dims(img, axis=0)
      st.image(img,width = 300)
      predict_x= model.predict(img)
      classes_x=np.argmax(predict_x,axis=1)
      numbers = [0,1,2,3,4,5]
      classes = ['Buildings', 'Forest', 'Glaciers', 'Mountains', 'Sea', 'Street']
      dir_clases = dict(zip(numbers,classes))
    with col2:
      st.write(f'Model is **{predict_x[0][classes_x[0]]*100:.2f}%** sure that it is **{dir_clases[classes_x[0]]}**')
      dir_clases = dict(zip(predict_x[0],classes))
      import collections
      od = collections.OrderedDict(sorted(dir_clases.items(),reverse=True))

      for key,values in od.items():
        st.write(f'{key*100:.2f}% - {values}')