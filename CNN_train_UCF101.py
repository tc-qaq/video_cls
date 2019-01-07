"""
Train on images split into directories. This assumes we've split our videos into frames and moved
them to their respective folders. Use keras 2+ and tensorflow 1+
"""
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from UCFdata import DataSet

data = DataSet()
checkpointer = ModelCheckpoint(
    filepath='./data/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)
early_stopper = EarlyStopping(patience=10)

tensorboard = TensorBoard(log_dir='./data/logs/')

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,        shear_range=0.2,        horizontal_flip=True,
        rotation_range=10.,    width_shift_range=0.2,  height_shift_range=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory('./data/train/',
        target_size=(299, 299),      batch_size=8,
        classes=data.classes,        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./data/test/',
        target_size=(299, 299),      batch_size=8,
        classes=data.classes,        class_mode='categorical')
    return train_generator, validation_generator

def get_model(weights='imagenet'):
    base_model = InceptionV3(weights=weights, include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(data.classes), activation='softmax')(x)
    
    model = Model(inputs=base_model.input,outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.complie(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def fine_tune_inception_layer(model):
    """After we fine-tune the dense layers, train deeper."""
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True
        
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(train_generator,        steps_per_epoch=100,
                        validation_data = validation_generator,        validation_steps=10,
                        epochs = nb_epoch,        callbacks=callbacks)
    return model

def main(weights_file):
    model = get_model()
    generators = get_generators()
    if (weights_file is None):
        print("Training top layers")
        model = train_model(model, 10, generators)
    else:
        print("loading saved model %s." % weights_file)
        model.load_weights(weights_file)
    
    model = fine_tune_inception_layer(model)
    model = train_model(model, 1000, generators, [checkpointer, early_stopper, tensorboard])
    
if __name__ == '__main__':
    weights_file = None
    main(weights_file)