from keras.applications import DenseNet121 
from keras_vis.utils import plot_saliency 
from keras_vis.backend import plt
from keras_vis.utils import plot_cam
from keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
# Load the DenseNet121 model with the weights trained on ImageNet
model = model.load_weights('./checkpoint/best_task4_server_cut1_pool_model_checkpoint.h5')

# Generate a random input image
images = ['coast/arnat59', 'forest/art114', 'highway/ar237', 'insidecity/a0004', 'mountain/ar1131', 'Opencountry/art582', 'street/art256', 'tallbuilding/a212017']
for name_img in images:
    img = image.load_img('./MIT_small_train_1/'+name_img+'.jpg', target_size= (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Compute the saliency map
    saliency = plot_saliency(model, x, class_id=None, layer_id=-1, filter_indices=None)
    # Plot the saliency map
    plt.imshow(saliency, cmap='jet') 
    # cut the name of image
    i = name_img.split('/')[1]
    plt.savefig(i+'.png')
    cam = plot_cam(model, x, class_id=None, layer_id=-1)
    plt.savefig(i+'_cam.png')