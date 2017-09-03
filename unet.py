from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 15
IMAGE_SIZE = 224


def u_net(image,phase_train):
	with tf.variable_scope("u_net"):
		w1_1=utils.weight_variable([3,3,int(image.shape[3]),32],name="w1_1")
		b1_1=utils.bias_variable([32],name="b1_1")
		conv1_1=utils.conv2d_basic(image,w1_1,b1_1)
		relu1_1 = tf.nn.relu(conv1_1, name="relu1_1")
		w1_2=utils.weight_variable([3,3,32,32],name="w1_2")
		b1_2=utils.bias_variable([32],name="b1_2")
		conv1_2=utils.conv2d_basic(relu1_1,w1_2,b1_2)
		relu1_2 = tf.nn.relu(conv1_2, name="relu1_2")
		pool1=utils.max_pool_2x2(relu1_2)
		bn1=utils.batch_norm(pool1,pool1.get_shape()[3],phase_train,scope="bn1")

		w2_1=utils.weight_variable([3,3,32,64],name="w2_1")
		b2_1=utils.bias_variable([64],name="b2_1")
		conv2_1=utils.conv2d_basic(bn1,w2_1,b2_1)
		relu2_1 = tf.nn.relu(conv2_1, name="relu2_1")
		w2_2=utils.weight_variable([3,3,64,64],name="w2_2")
		b2_2=utils.bias_variable([64],name="b2_2")
		conv2_2=utils.conv2d_basic(relu2_1,w2_2,b2_2)
		relu2_2 = tf.nn.relu(conv2_2, name="relu2_2")
		pool2=utils.max_pool_2x2(relu2_2)
		bn2=utils.batch_norm(pool2,pool2.get_shape()[3],phase_train,scope="bn2")

		w3_1=utils.weight_variable([3,3,64,128],name="w3_1")
		b3_1=utils.bias_variable([128],name="b3_1")
		conv3_1=utils.conv2d_basic(bn2,w3_1,b3_1)
		relu3_1 = tf.nn.relu(conv3_1, name="relu3_1")
		w3_2=utils.weight_variable([3,3,128,128],name="w3_2")
		b3_2=utils.bias_variable([128],name="b3_2")
		conv3_2=utils.conv2d_basic(relu3_1,w3_2,b3_2)
		relu3_2 = tf.nn.relu(conv3_2, name="relu3_2")
		pool3=utils.max_pool_2x2(relu3_2)
		bn3=utils.batch_norm(pool3,pool3.get_shape()[3],phase_train,scope="bn3")

		w4_1=utils.weight_variable([3,3,128,256],name="w4_1")
		b4_1=utils.bias_variable([256],name="b4_1")
		conv4_1=utils.conv2d_basic(bn3,w4_1,b4_1)
		relu4_1 = tf.nn.relu(conv4_1, name="relu4_1")
		w4_2=utils.weight_variable([3,3,256,256],name="w4_2")
		b4_2=utils.bias_variable([256],name="b4_2")
		conv4_2=utils.conv2d_basic(relu4_1,w4_2,b4_2)
		relu4_2 = tf.nn.relu(conv4_2, name="relu4_2")
		pool4=utils.max_pool_2x2(relu4_2)
		bn4=utils.batch_norm(pool4,pool4.get_shape()[3],phase_train,scope="bn4")

		w5_1=utils.weight_variable([3,3,256,512],name="w5_1")
		b5_1=utils.bias_variable([512],name="b5_1")
		conv5_1=utils.conv2d_basic(bn4,w5_1,b5_1)
		relu5_1 = tf.nn.relu(conv5_1, name="relu5_1")
		w5_2=utils.weight_variable([3,3,512,512],name="w5_2")
		b5_2=utils.bias_variable([512],name="b5_2")
		conv5_2=utils.conv2d_basic(relu5_1,w5_2,b5_2)
		relu5_2 = tf.nn.relu(conv5_2, name="relu5_2")
		bn5=utils.batch_norm(relu5_2,relu5_2.get_shape()[3],phase_train,scope="bn5")
		###up6
		W_t1 = utils.weight_variable([2, 2, 256, 512], name="W_t1")
		b_t1 = utils.bias_variable([256], name="b_t1")
		conv_t1 = utils.conv2d_transpose_strided(bn5, W_t1, b_t1, output_shape=tf.shape(relu4_2))
		merge1 = tf.concat([conv_t1,relu4_2],3)
		w6_1=utils.weight_variable([3,3,512,256],name="w6_1")
		b6_1=utils.bias_variable([256],name="b6_1")
		conv6_1=utils.conv2d_basic(merge1,w6_1,b6_1)
		relu6_1=tf.nn.relu(conv6_1, name="relu6_1")
		w6_2=utils.weight_variable([3,3,256,256],name="w6_2")
		b6_2=utils.bias_variable([256],name="b6_2")
		conv6_2=utils.conv2d_basic(relu6_1,w6_2,b6_2)
		relu6_2=tf.nn.relu(conv6_2, name="relu6_2")
		bn6=utils.batch_norm(relu6_2,relu6_2.get_shape()[3],phase_train,scope="bn6")
		###up7
		W_t2 = utils.weight_variable([2, 2, 128, 256], name="W_t2")
		b_t2 = utils.bias_variable([128], name="b_t2")
		conv_t2 = utils.conv2d_transpose_strided(bn6, W_t2, b_t2, output_shape=tf.shape(relu3_2))
		merge2 = tf.concat([conv_t2,relu3_2],3)
		w7_1=utils.weight_variable([3,3,256,128],name="w7_1")
		b7_1=utils.bias_variable([128],name="b7_1")
		conv7_1=utils.conv2d_basic(merge2,w7_1,b7_1)
		relu7_1=tf.nn.relu(conv7_1, name="relu7_1")
		w7_2=utils.weight_variable([3,3,128,128],name="w7_2")
		b7_2=utils.bias_variable([128],name="b7_2")
		conv7_2=utils.conv2d_basic(relu7_1,w7_2,b7_2)
		relu7_2=tf.nn.relu(conv7_2, name="relu7_2")
		bn7=utils.batch_norm(relu7_2,relu7_2.get_shape()[3],phase_train,scope="bn7")
		###up8
		W_t3 = utils.weight_variable([2, 2, 64, 128], name="W_t3")
		b_t3 = utils.bias_variable([64], name="b_t3")
		conv_t3 = utils.conv2d_transpose_strided(bn7, W_t3, b_t3, output_shape=tf.shape(relu2_2))
		merge3 = tf.concat([conv_t3,relu2_2],3)
		w8_1=utils.weight_variable([3,3,128,64],name="w8_1")
		b8_1=utils.bias_variable([64],name="b8_1")
		conv8_1=utils.conv2d_basic(merge3,w8_1,b8_1)
		relu8_1=tf.nn.relu(conv8_1, name="relu8_1")
		w8_2=utils.weight_variable([3,3,64,64],name="w8_2")
		b8_2=utils.bias_variable([64],name="b8_2")
		conv8_2=utils.conv2d_basic(relu8_1,w8_2,b8_2)
		relu8_2=tf.nn.relu(conv8_2, name="relu8_2")
		bn8=utils.batch_norm(relu8_2,relu8_2.get_shape()[3],phase_train,scope="bn8")
		###up9
		W_t4 = utils.weight_variable([2, 2, 32, 64], name="W_t4")
		b_t4 = utils.bias_variable([32], name="b_t4")
		conv_t4 = utils.conv2d_transpose_strided(bn8, W_t4, b_t4, output_shape=tf.shape(relu1_2))
		merge4 = tf.concat([conv_t4,relu1_2],3)
		w9_1=utils.weight_variable([3,3,64,32],name="w9_1")
		b9_1=utils.bias_variable([32],name="b9_1")
		conv9_1=utils.conv2d_basic(merge4,w9_1,b9_1)
		relu9_1=tf.nn.relu(conv9_1, name="relu9_1")
		w9_2=utils.weight_variable([3,3,32,32],name="w9_2")
		b9_2=utils.bias_variable([32],name="b9_2")
		conv9_2=utils.conv2d_basic(relu9_1,w9_2,b9_2)
		relu9_2=tf.nn.relu(conv9_2, name="relu9_2")
		bn9=utils.batch_norm(relu9_2,relu9_2.get_shape()[3],phase_train,scope="bn9")

		###output scoreMap
		w10=utils.weight_variable([1,1,32,NUM_OF_CLASSESS],name="w10")
		b10=utils.bias_variable([NUM_OF_CLASSESS],name="b10")
		conv10=utils.conv2d_basic(bn9,w10,b10)
		annotation_pred = tf.argmax(conv10, dimension=3, name="prediction")
		return annotation_pred,conv10


def inference(image, keep_prob,phase_train):

	mean_pixel = np.array([51386.56340487,102148.55162744,71218.35679277])

	processed_image = utils.process_image(image, mean_pixel)

	annotation_pred,conv10 = u_net(processed_image,phase_train)

	return tf.expand_dims(annotation_pred, dim=3), conv10


def train(loss_val, var_list):
	# optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
	optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
	grads = optimizer.compute_gradients(loss_val, var_list=var_list)
	if FLAGS.debug:
		# print(len(var_list))
		for grad, var in grads:
			utils.add_gradient_summary(grad, var)
	return optimizer.apply_gradients(grads)

def color_image(image, num_classes=10):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def main(argv=None):
	keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
	image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
	# phase_train=tf.placeholder(tf.bool)
	phase_train=tf.Variable(True,name="phase_train",trainable=False)
	annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
	pred_annotation, logits = inference(image, keep_probability,phase_train)
	##compute accuracy
	correct_pred=tf.equal(tf.cast(pred_annotation,tf.int32), annotation)
	accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
																		  labels=tf.squeeze(annotation, squeeze_dims=[3]),
																		  name="entropy")))

	trainable_var = tf.trainable_variables()
	skipLayers=[]
	var_list = [v for v in trainable_var if v.name.split('/')[1][0:7] not in skipLayers]
	if FLAGS.debug:
		for var in var_list:
			utils.add_to_regularization_and_summary(var)
	train_op = train(loss, var_list)

	print("Setting up summary op...")
	tf.summary.image("input_image", image, max_outputs=2)
	tf.summary.image("ground_truth", tf.cast(tf.image.grayscale_to_rgb(annotation), tf.float32), max_outputs=2)
	tf.summary.image("pred_annotation", tf.cast(tf.image.grayscale_to_rgb(pred_annotation), tf.float32), max_outputs=2)
	summary_op = tf.summary.merge_all()
	##two kinds of loss
	trainLoss_summary=tf.summary.scalar("entropy_train", loss)
	validationLoss_summary=tf.summary.scalar("entropy_validaton",loss)
	##two kinds of acc
	trainAcc_summary=tf.summary.scalar("acc_train", accuracy)
	valAcc_summary=tf.summary.scalar("acc_val", accuracy)

	print("Setting up image reader...")
	train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
	print(len(train_records))
	print(len(valid_records))

	print("Setting up dataset reader")
	image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
	if FLAGS.mode == 'train':
		train_dataset_reader = dataset.BatchDatset(train_records, image_options)
	validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

	sess = tf.Session()

	print("Setting up Saver...")
	saver = tf.train.Saver(trainable_var)
	summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

	sess.run(tf.global_variables_initializer())
	ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Model restored...")

	if FLAGS.mode == "train":
		for itr in xrange(MAX_ITERATION):
			train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
			valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
			feed_dict1 = {image: train_images, annotation: train_annotations,keep_probability: 0.85}
			feed_dict2={image: valid_images, annotation: valid_annotations,keep_probability: 0.85}
			sess.run(train_op, feed_dict=feed_dict1)

			if itr % 10 == 0:
				summary_merge = sess.run(summary_op, feed_dict=feed_dict2)
				train_loss, summary_train = sess.run([loss, trainLoss_summary], feed_dict=feed_dict1)
				val_loss,summary_val=sess.run([loss,validationLoss_summary],feed_dict=feed_dict2)
				train_acc,trainAccSumm=sess.run([accuracy,trainAcc_summary],feed_dict=feed_dict1)
				val_acc,valAccSumm=sess.run([accuracy,valAcc_summary],feed_dict=feed_dict2)
				print("Step: %d, Train_acc:%g" % (itr, train_acc))
				print("Step: %d, Val_acc:%g" % (itr, val_acc))
				print("==================>")
				summary_writer.add_summary(summary_train, itr)
				summary_writer.add_summary(summary_val, itr)
				summary_writer.add_summary(trainAccSumm, itr)
				summary_writer.add_summary(valAccSumm, itr)
				summary_writer.add_summary(summary_merge, itr)

			if itr % 500 == 0:
				valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
				valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
													   keep_probability: 1.0})
				print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
				saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

	elif FLAGS.mode == "visualize":
		valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
		pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
													keep_probability: 1.0})
		valid_annotations = np.squeeze(valid_annotations, axis=3)
		pred = np.squeeze(pred, axis=3)

		for itr in range(FLAGS.batch_size):
			utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
			utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
			utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
			print("Saved image: %d" % itr)


if __name__ == "__main__":
	tf.app.run()

