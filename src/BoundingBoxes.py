import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import cv2

data_dir = f"{os.getcwd()}/../data/"
iou_threshold = 1000
BATCH_SIZE = 64

class BoundingBoxes:
    def __init__(self):
        pass

    def draw_bounding_box_on_image(self, image, ymin, xmin, ymax, xmax, color=(255, 0, 0), thickness=5):
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

    def draw_bounding_boxes_on_image(self, image, boxes, color=[], thickness=5):
        boxes_shape = boxes.shape
        if not boxes_shape:
            return
        if len(boxes_shape) != 2 or boxes_shape[1] != 4:
            raise ValueError('Input must be of size [N, 4]')
        for i in range(boxes_shape[0]):
            self.draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3],
                                       boxes[i, 2], color[i], thickness)


    def draw_bounding_boxes_on_image_array(self, image, boxes, color=[], thickness=5):
        self.draw_bounding_boxes_on_image(image, boxes, color, thickness)
        return image

    def display_digits_with_boxes(self, images, pred_bboxes, bboxes, iou, title, bboxes_normalized=False):
        n = len(images)
        fig = plt.figure(figsize=(20, 4))
        plt.title(title)
        plt.yticks([])
        plt.xticks([])
        for i in range(n):
            ax = fig.add_subplot(1, 10, i + 1)
            bboxes_to_plot = []
            if (len(pred_bboxes) > i):
                bbox = pred_bboxes[i]
                bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1],
                        bbox[3] * images[i].shape[0]]
                bboxes_to_plot.append(bbox)
            if (len(bboxes) > i):
                bbox = bboxes[i]
                if bboxes_normalized == True:
                    bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1],
                            bbox[3] * images[i].shape[0]]
                bboxes_to_plot.append(bbox)
            img_to_draw = self.draw_bounding_boxes_on_image_array(image=images[i], boxes=np.asarray(bboxes_to_plot), color=[(255, 0, 0), (0, 255, 0)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img_to_draw)
            if len(iou) > i:
                color = "black"
                if (iou[i][0] < iou_threshold):
                    color = "red"
                ax.text(0.2, -0.3, "iou: %s" % (iou[i][0]), color=color, transform=ax.transAxes)
        plt.savefig(f"{data_dir}/figures/figure.png")

    def plot_metrics(self, metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0, ylim)
        # todo: define history
        plt.plot(history.history[metric_name], color='blue', label=metric_name)
        plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)

    def read_image_tfds(self, image, bbox):
        image = tf.cast(image, tf.float32)
        shape = tf.shape(image)
        factor_x = tf.cast(shape[1], tf.float32)
        factor_y = tf.cast(shape[0], tf.float32)
        image = tf.image.resize(image, (224, 224,))
        image = image / 127.5
        image -= 1
        bbox_list = [bbox[0] / factor_x,
                     bbox[1] / factor_y,
                     bbox[2] / factor_x,
                     bbox[3] / factor_y]

        return image, bbox_list

    def read_image_with_shape(self, image, bbox):
        original_image = image
        image, bbox_list = self.read_image_tfds(image, bbox)
        return original_image, image, bbox_list

    def read_image_tfds_with_original_bbox(self, data):
        image = data["image"]
        bbox = data["bbox"]
        shape = tf.shape(image)
        factor_x = tf.cast(shape[1], tf.float32)
        factor_y = tf.cast(shape[0], tf.float32)
        bbox_list = [bbox[1] * factor_x,
                     bbox[0] * factor_y,
                     bbox[3] * factor_x,
                     bbox[2] * factor_y]
        return image, bbox_list

    def dataset_to_numpy_util(self, dataset, batch_size=0, N=0):
        take_dataset = dataset.shuffle(1024)
        if batch_size > 0:
            take_dataset = take_dataset.batch(batch_size)
        if N > 0:
            take_dataset = take_dataset.take(N)
        if tf.executing_eagerly():
            ds_images, ds_bboxes = [], []
            for images, bboxes in take_dataset:
                ds_images.append(images.numpy())
                ds_bboxes.append(bboxes.numpy())
        return (np.array(ds_images), np.array(ds_bboxes))

    def dataset_to_numpy_with_original_bboxes_util(self, dataset, batch_size=0, N=0):
        normalized_dataset = dataset.map(self.read_image_with_shape)
        if batch_size > 0:
            normalized_dataset = normalized_dataset.batch(batch_size)
        if N > 0:
            normalized_dataset = normalized_dataset.take(N)
        if tf.executing_eagerly():
            ds_original_images, ds_images, ds_bboxes = [], [], []
        for original_images, images, bboxes in normalized_dataset:
            ds_images.append(images.numpy())
            ds_bboxes.append(bboxes.numpy())
            ds_original_images.append(original_images.numpy())
        return np.array(ds_original_images), np.array(ds_images), np.array(ds_bboxes)

    def get_visualization_training_dataset(self):
        dataset, info = tfds.load("caltech_birds2010", split="train", with_info=True, data_dir=data_dir, download=False)
        print(info)
        visualization_training_dataset = dataset.map(self.read_image_tfds_with_original_bbox, num_parallel_calls=16)
        return visualization_training_dataset

    def get_visualization_validation_dataset(self):
        dataset = tfds.load("caltech_birds2010", split="test", data_dir=data_dir, download=False)
        visualization_validation_dataset = dataset.map(self.read_image_tfds_with_original_bbox, num_parallel_calls=16)
        return visualization_validation_dataset

    def get_training_dataset(self, dataset):
        dataset = dataset.map(self.read_image_tfds, num_parallel_calls=16)
        dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(-1)
        return dataset

    def get_validation_dataset(self, dataset):
        dataset = dataset.map(self.read_image_tfds, num_parallel_calls=16)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.repeat()
        return dataset

    def feature_extractor(self, inputs):
        mobilenet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        feature_extractor = mobilenet_model(inputs)
        return feature_extractor

    def dense_layers(self, features):
        x = tf.keras.layers.GlobalAveragePooling2D()(features)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        return x

    def bounding_box_regression(self, x):
        bounding_box_regression_output = tf.keras.layers.Dense(units=4, activation='relu', name="bounding_box")(x)
        return bounding_box_regression_output

    def final_model(self, inputs):
        feature_cnn = self.feature_extractor(inputs)
        last_dense_layer = self.dense_layers(feature_cnn)
        bounding_box_output = self.bounding_box_regression(last_dense_layer)
        model = tf.keras.Model(inputs=inputs, outputs=bounding_box_output)
        return model

    def define_and_compile_model(self):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        model = self.final_model(inputs)
        model.compile()
        return model

if __name__ == '__main__':
    bounding_boxes = BoundingBoxes()
    plt.rc('image', cmap='gray')
    plt.rc('grid', linewidth=0)
    plt.rc('xtick', top=False, bottom=False, labelsize='large')
    plt.rc('ytick', left=False, right=False, labelsize='large')
    plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
    plt.rc('text', color='a8151a')
    plt.rc('figure', facecolor='F0F0F0')
    MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")
    visualization_training_dataset = bounding_boxes.get_visualization_training_dataset()
    visualization_validation_dataset = bounding_boxes.get_visualization_validation_dataset()
    visualization_training_images, visualization_training_bboxes = bounding_boxes.dataset_to_numpy_util( visualization_training_dataset, N=10)
    bounding_boxes.display_digits_with_boxes(np.array(visualization_training_images), np.array([]),
                                             np.array(visualization_training_bboxes), np.array([]), "training images and their bboxes")
    training_dataset = bounding_boxes.get_training_dataset(visualization_training_dataset)
    validation_dataset = bounding_boxes.get_validation_dataset(visualization_validation_dataset)
    model = bounding_boxes.define_and_compile_model()
    model.summary()