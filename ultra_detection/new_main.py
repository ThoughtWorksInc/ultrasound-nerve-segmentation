import os
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python.training import queue_runner
from ultra_detection.model import inference

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'artifacts/logs',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_string('model_dir', 'artifacts/models',
                           """Directory where to save models """)
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of samples per batch.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20000


class Trainer:
  def __init__(self, input_path, sess, batch_size):
    self.batch_size = batch_size
    self.learning_rate = 1e-3
    self.input_path = input_path
    self.base_names = [
      os.path.splitext(f)[0]
      for f in os.listdir(self.input_path)
      if f != '.DS_Store' and 'mask' not in f
      ]
    self.sess = sess
    self.file_ext = '.png'

  def _generate_reader(self, base_name_tensor, name_processor=lambda x: x):
    tensor = tf.convert_to_tensor(base_name_tensor)
    name_queue = tf.train.string_input_producer(
      self.input_path + '/' + name_processor(tensor) + self.file_ext,
      shuffle=False
    )
    reader = tf.WholeFileReader()
    return reader.read(name_queue)

  def load(self):
    image_key, image_file = self._generate_reader(self.base_names)
    mask_key, mask_file = self._generate_reader(self.base_names, lambda x: x + '_mask')
    image = tf.image.decode_png(image_file, channels=1)
    mask = tf.image.decode_png(mask_file)

    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    image = tf.image.per_image_whitening(image)
    mask /= 255.0

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    num_preprocess_threads = 16
    image_batch, mask_batch = tf.train.shuffle_batch(
      [image, mask],
      batch_size=self.batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * self.batch_size,
      min_after_dequeue=min_queue_examples,
      shapes=[[320, 320, 1], [320, 320, 1]]
    )

    return image_batch, mask_batch

  def loss(self, mask, mask_infer):
    return tf.nn.l2_loss(mask_infer - mask)

  def train(self, loss):
    tf.scalar_summary(loss.op.name, loss)
    # solver
    return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

  def train_step(self, image, mask):
    mask_infer = inference(image, 320, 320)
    loss = self.loss(mask, mask_infer)
    return self.train(loss), loss

  def test(self):
    pass


def train():
  with tf.Session() as sess:
    trainer = Trainer('../splitted_png', sess, FLAGS.batch_size)

    image, mask = trainer.load()
    train_op, loss_op = trainer.train_step(image, mask)

    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    sess.run(init)

    coord = tf.train.Coordinator()

    queue_threads = queue_runner.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

    for step in range(FLAGS.max_steps):
      if coord.should_stop():
        break

      start_time = time.time()
      _, loss_value = sess.run([train_op, loss_op])
      duration = time.time() - start_time

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

    coord.request_stop()
    coord.join(queue_threads)


if __name__ == '__main__':
  train()
