#Imports. Loading the Data. 
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate



# Setting the paths to the data
base_dir = "/Users/irenzo/Desktop/test_task"
train_dir = os.path.join(base_dir, "train_v2")
test_dir = os.path.join(base_dir, "test_v2")
train_csv_path = os.path.join(base_dir, "train_ship_segmentations_v2.csv")
test_csv_path = os.path.join(base_dir, "sample_submission_v2.csv")

# Loading the CSV files
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Definning Model Architecture
def unet(input_size=(768, 768, 3)):
    inputs = Input(input_size)
    # Downsample
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    # Bottleneck
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    # Upsample
    u4 = UpSampling2D((2, 2))(c3)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    u5 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(u5)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Compiling Model
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def rle_decode(mask_rle, shape=(768, 768)):
    if pd.isnull(mask_rle):
        return np.zeros(shape)
    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape).T

batch_size = 10

def image_mask_generator(image_ids, batch_size, base_dir, dataframe):
    while True:
        batch_ids = np.random.choice(image_ids, batch_size)
        batch_images = []
        batch_masks = []
        for image_id in batch_ids:
            image_path = os.path.join(base_dir, "train_v2", image_id)
            image = Image.open(image_path)
            image_np = np.array(image) / 255.0
            
            rle_masks = dataframe[dataframe['ImageId'] == image_id]['EncodedPixels']
            mask = np.zeros((768, 768))
            for rle_mask in rle_masks.dropna():
                mask += rle_decode(rle_mask)
            mask = np.clip(mask, 0, 1)
            
            batch_images.append(image_np)
            batch_masks.append(mask.reshape(768, 768, 1))
        
        yield np.array(batch_images), np.array(batch_masks)


# Model Training
#I trained this model only on 1% from all data since it was time and resource consuming
train_df_sampled = train_df.sample(frac=0.01, random_state=42)

gen_sampled = image_mask_generator(train_df_sampled['ImageId'].unique(), batch_size, base_dir, train_df_sampled)

steps_per_epoch_sampled = len(train_df_sampled) // batch_size

epochs = 1  

model.fit(gen_sampled, steps_per_epoch=steps_per_epoch_sampled, epochs=epochs)

# Saving Model
#model.save('my_ship_det_model.keras')