"""
Classify a few images through our CNN.
"""
import numpy as np, operator, random, glob
from UCFdata import DataSet
from processer import process_image
from keras.models import load_model

def main(nb_images=5): 
    data = DataSet()
    model = load_model('data/checkpoints/inception.057-1.16.hdf5')
    # Get all our test images.
    images = glob.glob('./data/test/**/*.jpg')
    for _ in range(nb_images):
        print('-', *80)
        sample = random.randint(0, len(images) - 1)
        image = images[sample]
        print(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)
        predictions = model.predict(image_arr)
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]
        
        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1

if __name__ == '__main__':
    main()      