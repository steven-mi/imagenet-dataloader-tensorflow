# imagenet-dataloader-tensorflow
imagenet-dataloader for tensorflow

**original repositorys**
- https://github.com/tensorflow/models/tree/master/research/inception/inception/data
- https://github.com/ischlag/tensorflow-input-pipelines

## Getting Started
This will download the imagenet dataset
```bash
git clone https://github.com/steven-mi/imagenet-dataloader-tensorflow.git
cd downloader
chmod u+x run_me.sh
./run_me.sh
```
## Example
example.py
```python
import tensorflow as tf
import time

sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.imagenet import imagenet_data
  d = imagenet_data(batch_size=64, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

for i in range(10):
  print("batch ", i)
  image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
  print(image_batch.shape)
  print(target_batch.shape)
print("done!")

print("Closing the queue and the session. This will lead to the following warning/error ...")
time.sleep(8)
d.close()
sess.close()
exit()
```
