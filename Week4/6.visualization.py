import tensorflow as tf
from keras import Model
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from tf_explain.core import OcclusionSensitivity, ExtractActivations
from tf_explain.core.grad_cam import GradCAM

model = tf.keras.models.load_model('./checkpoint/best_task4_server_cut1_pool_model_checkpoint.h5')
model.summary()
# Generate a random input image
images = ['coast/arnat59', 'forest/art114', 'highway/a836030', 'inside_city/a0004', 'mountain/art1131',
          'Opencountry/art582', 'street/art256', 'tallbuilding/a212017']
for name_img in images:
    img = tf.keras.preprocessing.image.load_img('./MIT_small_train_1/test/' + name_img + '.jpg', target_size=(224, 224), )
    x = tf.keras.preprocessing.image.img_to_array(img)
    data = ([x], None)
    explainer = GradCAM()
    grid = explainer.explain(data, model, layer_name='conv4_block16_1_conv', class_index=0)
    i = name_img.split('/')[1]
    explainer.save(grid, output_dir='./XAI/', output_name=i + '_cam.png')
    """
    explainer = OcclusionSensitivity()
    grid = explainer.explain(data, model, class_index=0, patch_size=1)
    explainer.save(grid, output_dir='./XAI/', output_name=i + '_occlusion.png')

    explainer = ExtractActivations()
    grid = explainer.explain(data, model, ['conv4_block16_1_conv'])
    explainer.save(grid, output_dir='./XAI/', output_name=i + '_activation.png')
    """
    model = Model(inputs=model.input, outputs=model.get_layer('conv4_block16_1_conv').output)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    # save the feature map
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(features[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.savefig('./XAI/' + i + '_feature.png')
