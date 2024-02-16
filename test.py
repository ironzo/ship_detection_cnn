# Imports

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from train import rle_decode, batch_size, test_df, base_dir, model

# Test Model
def test_image_mask_generator(image_ids, base_dir, dataframe):
    for image_id in image_ids:
        image_path = os.path.join(base_dir, "test_v2", image_id)
        image = Image.open(image_path)
        image_np = np.array(image) / 255.0
        
        rle_masks = dataframe[dataframe['ImageId'] == image_id]['EncodedPixels']
        mask = np.zeros((768, 768))
        for rle_mask in rle_masks.dropna():
            mask += rle_decode(rle_mask)
        mask = np.clip(mask, 0, 1)
        
        yield image_np[np.newaxis, ...], mask.reshape(1, 768, 768, 1)

# Creating the test generator
test_gen = test_image_mask_generator(test_df['ImageId'].unique(), base_dir, test_df)

# Calculating the number of steps for the test data
test_steps = len(test_df) // batch_size

# Evaluating the model
test_loss, test_accuracy = model.evaluate(test_gen, steps=test_steps)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

dice_scores = []
for i in range(test_steps):
    x, y_true = next(test_gen)
    y_pred = model.predict(x)
    dice_scores.append(dice_coefficient(y_true, y_pred))

average_dice_score = np.mean(dice_scores)
print(f"Average Dice Score: {average_dice_score}")

# Visualizing some test images with their predicted masks
for i in range(3): 
    x, y_true = next(test_gen)
    y_pred = model.predict(x)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(x[0])
    plt.title("Test Image")
    plt.subplot(1, 3, 2)
    plt.imshow(y_true[0].squeeze(), cmap='gray')
    plt.title("True Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[0].squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.show()

    