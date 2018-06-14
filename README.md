# TensorFlow MirroredStrategy sample

`MirroredStrategy` sample using MNIST.

- https://www.youtube.com/watch?v=bRMGoPqsn20
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute
- https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/MirroredStrategyG

## Setup
```bash
virtualenv env
source env/bin/activate
pip install tensorflow
```

## Introduction of MirroredStrategy
`MirroredStrategy` is enabled through a `train_distribute` argument in `tf.estimator.RunConfig`.

```python
distribution = tf.contrib.distribute.MirroredStrategy()

config = tf.estimator.RunConfig(
  train_distribute=distribution,
  model_dir="/tmp/mnist_convnet_model")

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    config=config)
```

As stated in the official example, batch size needs to be divide by the number of GPUs as of this version.  
In this code, `per_device_batch_size()` defined in `models/official/utils/misc/distribution_utils.py` is used accordingly.

- https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py
- https://github.com/tensorflow/models/blob/master/official/utils/misc/distribution_utils.py

## Sample Result

The following table shows the result of a calculation obtained with `GeForce GTX 1080`.

|  # of GPUs  | Batch Size | Accuracy   |  Time (for 10000 steps) |
| ----------- | ---------- | ---------- | ----------------------- |
|  1          | 100        | 0.9511     |  68.0 sec               |
|  2          | 50         | 0.9515     |  90.6 sec               |
