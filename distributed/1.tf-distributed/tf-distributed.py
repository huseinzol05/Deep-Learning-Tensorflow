import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags

flags.DEFINE_string('job_name', 'ps', 'job type: worker or ps')
flags.DEFINE_integer(
    'task_index',
    0,
    'Worker task index, should be >= 0. task_index=0 is '
    'the chief worker task the performs the variable '
    'initialization',
)

flags.DEFINE_string(
    'ps_hosts', 'localhost:2222', 'Comma-separated list of hostname:port pairs'
)
flags.DEFINE_string(
    'worker_hosts',
    'localhost:2223,localhost:2224,localhost:2225',
    'Comma-separated list of hostname:port pairs',
)

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


def main(_):
    mnist = input_data.read_data_sets('/tmp/mnist-data', one_hot = True)
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    cluster_spec = tf.train.ClusterSpec(
        {
            'ps': FLAGS.ps_hosts.split(','),
            'worker': FLAGS.worker_hosts.split(','),
        }
    )
    server = tf.train.Server(
        cluster_spec, job_name = FLAGS.job_name, task_index = FLAGS.task_index
    )
    if FLAGS.job_name == 'ps':
        server.join()

    worker_device = '/job:worker/task:{}'.format(FLAGS.task_index)
    device, target = (
        tf.train.replica_device_setter(
            worker_device = worker_device, cluster = cluster_spec
        ),
        server.target,
    )
    with tf.device(device):
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        X = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        Y = tf.placeholder(tf.float32, [None, 10])
        forward = tf.layers.dense(X, 200, activation = tf.nn.relu)
        logits = tf.layers.dense(forward, 10)
        # variable_summaries(logits)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logits)
        )
        tf.summary.scalar('training_loss', loss)
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(
            loss, global_step = global_step
        )

        def run_train_epoch(target, FLAGS, epoch_index):
            train_step = 100 * (epoch_index + 1)
            hooks = [tf.train.StopAtStepHook(last_step = train_step)]
            i = 0
            with tf.train.MonitoredTrainingSession(
                master = target,
                is_chief = (FLAGS.task_index == 0),
                checkpoint_dir = 'test-tf-distributed',
                hooks = hooks,
            ) as sess:
                while not sess.should_stop():
                    batch_x, batch_y = mnist.train.next_batch(32)
                    train_feed = {X: batch_x, Y: batch_y}
                    global_loss, _ = sess.run(
                        [loss, optimizer], feed_dict = train_feed
                    )
                    print('iteration %d, loss %f' % (i + 1, global_loss))
                    i += 1

        for e in range(5):
            run_train_epoch(target, FLAGS, e)


tf.app.run()
