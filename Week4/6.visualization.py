import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM

model = tf.keras.models.load_model('./checkpoint/best_task4_server_cut1_pool_model_checkpoint.h5')
model.summary()
# Generate a random input image
images = ['coast/arnat59', 'forest/art114', 'highway/a836030', 'inside_city/a0004', 'mountain/art1131',
          'Opencountry/art582', 'street/art256', 'tallbuilding/a212017']
for name_img in images:
    img = tf.keras.preprocessing.image.load_img('./MIT_small_train_1/test/' + name_img + '.jpg', target_size=(224, 224), )
    img = tf.keras.preprocessing.image.img_to_array(img)
    data = ([img], None)
    explainer = GradCAM()
    grid = explainer.explain(data, model, layer_name='conv4_block16_1_conv', class_index=0)
    i = name_img.split('/')[1]
    explainer.save(grid, output_dir='./XAI/', output_name=i + '.png')
