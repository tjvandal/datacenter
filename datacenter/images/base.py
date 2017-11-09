import os, sys
import scipy.misc
import numpy as np
import tensorflow as tf
import requests, zipfile
import StringIO
from PIL import Image

def _download_images(dir):
    url = 'https://github.com/jbhuang0604/SelfExSR/archive/master.zip'
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall()
    for name in z.namelist():
        if '.png' in name:
            to_file = os.path.join(dir, name)
            if not os.path.exists(os.path.dirname(to_file)):
                os.makedirs(os.path.dirname(to_file))
            img = Image.open(StringIO.StringIO(z.read(name)))
            img.save(os.path.join(dir, name))

class SuperResData:
    def __init__(self, upscale_factor=2, imageset='Set5'):
        self.upscale_factor = upscale_factor
        self.imageset = imageset
        self._base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if not os.path.exists(self._base_dir):
            _download_images(self._base_dir)
        self.data_dir = os.path.join(self._base_dir, 'SelfExSR-master/data/', imageset,
                                     'image_SRF_%i' % self.upscale_factor)

    def read(self):
        hr_images = {}
        lr_images = {}
        for i, f in enumerate(os.listdir(self.data_dir)):
            img = scipy.misc.imread(os.path.join(self.data_dir, f))
            if "HR" in f:
                hr_images["".join(f.split("_")[:3])] = img
            elif "LR" in f:
                lr_images["".join(f.split("_")[:3])] = img
        lr_keys = sorted(lr_images.keys())
        hr_keys = sorted(hr_images.keys())
        assert lr_keys == hr_keys
        for k in hr_keys:
            yield lr_images[k], hr_images[k]

    def make_patches(self, patch_size=15, stride=8):
        """
        Args:
            patch_size: size of low-resolution subimages
            stride: step length between subimages
        """
        X_sub = []
        Y_sub = []
        for x, y in self.read():
            if len(x.shape) != 3:
                continue
            h, w, _ = x.shape
            for i in np.arange(0, h, stride):
                for j in np.arange(0, w, stride):
                    hi_low, hi_high = i, i+patch_size
                    wi_low, wi_high = j, j+patch_size
                    if (hi_high > h) or (wi_high > w):
                        continue
                    X_sub.append(x[np.newaxis,hi_low:hi_high,wi_low:wi_high])
                    Y_sub.append(y[np.newaxis,2*hi_low:2*hi_high,2*wi_low:2*wi_high])
        X_sub = np.concatenate(X_sub, axis=0)
        Y_sub = np.concatenate(Y_sub, axis=0)
        return X_sub, Y_sub


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process_image(img, upsample=2, stride=15, imsize=31, is_training=True):
    inputs, labels = [], []
    h, w, d = img.shape
    imgsub = img[:upsample*int(h/upsample), :upsample*int(w/upsample)]
    #img_l = cv2.resize(imgsub, (0,0), fx=1./upsample, fy=1./upsample, interpolation=cv2.INTER_CUBIC)
    #imgup = cv2.resize(img_l, (0,0), fx=1.*upsample, fy=1.*upsample, interpolation=cv2.INTER_CUBIC)
    img_l = scipy.misc.imresize(imgsub, 1./upsample)
    imgup = scipy.misc.imresize(img_l, float(upsample))

    imgsub = imgsub[upsample:-upsample, upsample:-upsample]
    imgup = imgup[upsample:-upsample, upsample:-upsample]
    h, w, d = imgsub.shape # reset with the new dimensions

    if not is_training:
        return imgup[np.newaxis], imgsub[np.newaxis]

    for y in np.arange(0, h, stride):
        for x in np.arange(0, w, stride):
            ylow, yhigh = y, y+imsize
            xlow, xhigh = x, x+imsize
            if (xhigh > w) or (yhigh > h):
                continue

            labels += [imgsub[np.newaxis, ylow:yhigh, xlow:xhigh]]
            inputs += [imgup[np.newaxis, ylow:yhigh, xlow:xhigh, :]]

    return np.concatenate(inputs, axis=0), np.concatenate(labels, axis=0)

def write_records(inputs, labels, write_dir, name):
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    writer = tf.python_io.TFRecordWriter(os.path.join(write_dir, name + '.tfrecords'))
    num_examples = len(inputs)
    for index in range(num_examples):
        n, height, width, depth = inputs[index].shape
        for j in range(n):
            x_in = inputs[index][j].astype(np.float32).tostring()
            lab = labels[index][j].astype(np.float32).tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(lab),
                'image': _bytes_feature(x_in),
                'depth': _int64_feature(depth),
                'height': _int64_feature(height),
                'width': _int64_feature(width)
                }))
            writer.write(example.SerializeToString())
    writer.close()

def build_dataset(filelist, is_training=True):
    file_count = 0
    X, Y = [], []
    tffile_dir = os.path.dirname(filelist[0]) + "_tfrecords_%i" % upsample
    for j, f in enumerate(filelist):
        img = scipy.misc.imread(f, mode='RGB')
        inputs, labels = process_image(img, is_training=is_training, w_matlab=False)
        X.append(inputs)
        Y.append(labels)

        if (j % chunksize == 0) and (j!=0):
            if is_training:
                write_records(X, Y, tffile_dir, 'train_%i' % file_count)
            else:
                write_records(X, Y, tffile_dir, 'test_%i' % file_count)
            print("Training:", is_training, " File number:", file_count)
            file_count += 1
            X, Y = [], []

    if is_training:
        write_records(X, Y, tffile_dir, 'train_%i' % file_count)
    else:
        write_records(X, Y, tffile_dir, 'test_%i' % file_count)

def read_and_decode(filename_queue, is_training=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.int64),
          'height':tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64)
      })

    with tf.device("/cpu:0"):
        if is_training:
            imgshape = [FLAGS.input_size, FLAGS.input_size, FLAGS.depth]
        else:
            depth = tf.cast(tf.reshape(features['depth'], []), tf.int32)
            width = tf.cast(tf.reshape(features['width'], []), tf.int32)
            height = tf.cast(tf.reshape(features['height'], []), tf.int32)
            imgshape = tf.stack([height, width, depth])

        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, imgshape)

        label = tf.decode_raw(features['label'], tf.float32)
        label = tf.reshape(label, imgshape)
        label_y = imgshape[0] - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
        label_x = imgshape[1] - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
        label = tf.slice(label, [FLAGS.padding, FLAGS.padding, 0], [label_y, label_x, -1])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(label, tf.float32) * (1. / 255) - 0.5
        return image, label

def read_tf(data_dir, train, batch_size, num_epochs=None):
    '''
    Specify directory of which to read from, if it's training, and batch_size
    Set: ['Set5', 'Set14']
    '''
    if train:
        files = [os.path.join(FLAGS.data_dir, f) for f in os.listdir(FLAGS.data_dir) if 'train' in f]
    else:
        files = [os.path.join(FLAGS.test_dir, f) for f in os.listdir(FLAGS.test_dir)]
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            files, num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue, is_training=train)

        # Shuffle the examples and collect them into batch_size batches.
        # We run this in two threads to avoid being a bottleneck.
        if train:
            image, label = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=8,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        else:
            image = tf.expand_dims(image, 0)
            label = tf.expand_dims(label, 0)
        return image,  label



if __name__ == '__main__':
    d = SuperResData(imageset='Set5')
    d.make_patches()
