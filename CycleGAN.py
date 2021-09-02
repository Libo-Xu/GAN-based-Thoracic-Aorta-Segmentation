import tensorflow as tf
from tensorflow_addons.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from collections import OrderedDict

import numpy as np
import random
import datetime
import time
import json
import csv
import sys
import os
import skimage.io as io


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
np.random.seed(seed=42)




class CycleGAN:
    def __init__(self, lr_D=2e-4, lr_G=2e-4, image_shape=(256, 256, 1),
                 date_time_string_addition='_test'):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_cycle = 10.0  # Cyclic loss weight A_2_B
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.lambda_seg = 1.0  # weight for segmentation loss in end-to-end training
        self.lambda_id = 0.1 * self.lambda_cycle
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.epochs = 100  # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 50

        # Decide to train u-net and
        self.use_end2end_training = True

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True


        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = False

        # Tweaks
        self.REAL_LABEL = 0.9  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Discriminator model ==========

        loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        # D_A.summary()

        # Build discriminators
        self.D_A = self.modelDiscriminator(name='D_A_model')
        self.D_B = self.modelDiscriminator(name='D_B_model')
        # plot_model(self.D_A, to_file='D_A.png', show_shapes=True)
        # plot_model(self.D_B, to_file='D_B.png', show_shapes=True)
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)


        # ======= Generator model ==========
        # segmentor
        self.seg_A = self.get_UNet(Base=32, depth=4, inc_rate=2,
                                   activation='relu', drop=False, batchnorm=True, N=2, name='seg_A')
        # Generators
        self.G_A2B = self.modelGenerator(name='G_A2B_model')
        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        # plot_model(self.G_A2B, to_file='G_A2B.png', show_shapes=True)
        # plot_model(self.G_B2A, to_file='G_B2A.png', show_shapes=True)
        # plot_model(self.seg_A, to_file='seg.png', show_shapes=True)

        # Generator builds
        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        segment_realA = self.seg_A(real_A)
        segment_fakeA = self.seg_A(synthetic_A)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)
        id_A = self.G_B2A(real_A)
        id_B = self.G_A2B(real_B)
        self.D_A.trainable = False
        self.D_B.trainable = False
        dA_guess_synthetic = self.D_A(synthetic_A)
        dB_guess_synthetic = self.D_B(synthetic_B)

        if self.use_end2end_training:
            model_outputs = [reconstructed_A, reconstructed_B, segment_realA, segment_fakeA, dA_guess_synthetic,
                             dB_guess_synthetic, id_A, id_B]
            compile_losses = [self.cycle_loss, self.cycle_loss, self.dice_coef_loss, self.dice_coef_loss, self.lse,
                              self.lse, self.cycle_loss, self.cycle_loss]
            compile_weights = [self.lambda_cycle, self.lambda_cycle, self.lambda_seg, self.lambda_seg,
                               self.lambda_D, self.lambda_D, self.lambda_id, self.lambda_id]
        else:
            model_outputs = [reconstructed_A, reconstructed_B, dA_guess_synthetic, dB_guess_synthetic, id_A, id_B]
            compile_losses = [self.cycle_loss, self.cycle_loss, self.lse, self.lse, self.cycle_loss, self.cycle_loss]
            compile_weights = [self.lambda_cycle, self.lambda_cycle, self.lambda_D, self.lambda_D,
                               self.lambda_id, self.lambda_id]

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

        # ======= Data ==========
        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        data = self.load_data()

        self.A_train = data["trainA_images"]
        self.B_train = data["trainB_images"]
        self.A_mask = data["maskA_images"]
        self.B_mask = data["maskB_images"]
        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]


        if not self.use_data_generator:
            print('Data has been loaded')

        # ======= Create designated run folder and store meta data ==========
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writeMetaDataToJSON()

        # ======= Avoid pre-allocating GPU memory ==========
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.Session(config=config)


        # ======= Initialize training ==========
        sys.stdout.flush()
        # plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
        # self.load_model_and_generate_synthetic_images()

    # ===============================================================================
    # Training
    def train(self, epochs, batch_size=1, save_interval=1):
        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
            # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)

                DA_loss = DA_loss_real + DA_loss_synthetic
                DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            if self.use_end2end_training:
                target_data = [real_images_A, real_images_B, masks_A, masks_B,
                               ones, ones, real_images_A, real_images_B]
            else:
                target_data = [real_images_A, real_images_B, ones, ones, real_images_A, real_images_B]



            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(x=[real_images_A, real_images_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            reconstruction_loss_A = G_loss[1]
            reconstruction_loss_B = G_loss[2]
            segA_dice_loss = G_loss[3]
            segB_dice_loss = G_loss[4]
            gA_d_loss_synthetic = G_loss[5]
            gB_d_loss_synthetic = G_loss[6]
            id_loss_A = G_loss[7]
            id_loss_B = G_loss[8]

            # Identity training
            #             if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
            #                 G_A2B_identity_loss = self.G_A2B.train_on_batch(x=real_images_B, y=real_images_B)
            #                 G_B2A_identity_loss = self.G_B2A.train_on_batch(x=real_images_A, y=real_images_A)
            #                 print('G_A2B_identity_loss:', G_A2B_identity_loss)
            #                 print('G_B2A_identity_loss:', G_B2A_identity_loss)



            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)

            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            segA_dice_losses.append(segA_dice_loss)
            segB_dice_losses.append(segB_dice_loss)

            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)

            id_A_losses.append(id_loss_A)
            id_B_losses.append(id_loss_B)

            D_losses.append(D_loss)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)

            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            seg_loss = segA_dice_loss + segB_dice_loss
            seg_losses.append(seg_loss)

            id_loss = id_loss_A + id_loss_B
            id_losses.append(id_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('segA_loss:', segA_dice_loss)
            print('segB_loss:', segB_dice_loss)
            print('reconstruction_loss: ', reconstruction_loss)
            print('DA_loss:', DA_loss)
            print('DB_loss:', DB_loss)
            print('id_loss', id_loss_A)

            if loop_index % 10 == 0:
                # Save temporary images continously
                self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        segA_dice_losses = []
        segB_dice_losses = []
        seg_losses = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []
        id_A_losses = []
        id_B_losses = []
        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []
        id_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # self.saveImages('(init)')

        # labels
        label_shape = (batch_size,) + self.D_A.output_shape[1:]
        ones = np.ones(shape=label_shape) * self.REAL_LABEL
        zeros = ones * 0

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    real_images_A = images[0]
                    real_images_B = images[1]
                    if len(real_images_A.shape) == 3:
                        real_images_A = real_images_A[:, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, np.newaxis]

                    # Run all training steps
                    run_training_iteration(loop_index, self.data_generator.__len__())

                    # Store models
                    if loop_index % 10000 == 0:
                        self.saveModel(self.D_A, loop_index)
                        self.saveModel(self.D_B, loop_index)
                        self.saveModel(self.G_A2B, loop_index)
                        self.saveModel(self.G_B2A, loop_index)

                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__():
                        break

                    loop_index += 1

            else:  # Train with all data in cache
                A_train = self.A_train
                B_train = self.B_train
                A_mask = self.A_mask
                B_mask = self.B_mask
                random_order_A = np.random.randint(len(A_train), size=len(A_train))
                random_order_B = np.random.randint(len(B_train), size=len(B_train))
                epoch_iterations = max(len(random_order_A), len(random_order_B))
                min_nr_imgs = min(len(random_order_A), len(random_order_B))

                # If we want supervised learning the same images form
                # the two domains are needed during each training iteration
                for loop_index in range(0, epoch_iterations, batch_size):
                    if loop_index + batch_size >= min_nr_imgs:
                        # If all images soon are used for one domain,
                        # randomly pick from this domain
                        if len(A_train) <= len(B_train):
                            indexes_A = np.random.randint(len(A_train), size=batch_size)

                            # if all images are used for the other domain
                            if loop_index + batch_size >= epoch_iterations:
                                indexes_B = random_order_B[epoch_iterations - batch_size: epoch_iterations]
                            else:  # if not used, continue iterating...
                                indexes_B = random_order_B[loop_index: loop_index + batch_size]

                        else:  # if len(B_train) <= len(A_train)
                            indexes_B = np.random.randint(len(B_train), size=batch_size)
                            # if all images are used for the other domain
                            if loop_index + batch_size >= epoch_iterations:
                                indexes_A = random_order_A[epoch_iterations - batch_size: epoch_iterations]
                            else:  # if not used, continue iterating...
                                indexes_A = random_order_A[loop_index: loop_index + batch_size]

                    else:
                        indexes_A = random_order_A[loop_index: loop_index + batch_size]
                        indexes_B = random_order_B[loop_index: loop_index + batch_size]

                    sys.stdout.flush()
                    real_images_A = A_train[indexes_A]
                    real_images_B = B_train[indexes_B]
                    masks_A = A_mask[indexes_A]
                    masks_B = B_mask[indexes_B]
                    # Run all training steps
                    run_training_iteration(loop_index, epoch_iterations)

            # ================== within epoch loop end ==========================

            if epoch % save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch,
                      '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B)

            if epoch % 5 == 0:
                if self.use_end2end_training:
                    self.saveModel(self.seg_A, epoch)

                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)

            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'segA_dice_losses:': segA_dice_losses,
                'segB_dice_losses:': segB_dice_losses,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()

    def load_data(self):

        trainA_path = './trainA/'
        trainB_path = './trainB/'
        maskA_path = './maskA/'
        maskB_path = './maskB/'
        trainA_images = self.create_image_array(trainA_path, A=True)
        trainB_images = self.create_image_array(trainB_path)
        maskA_images = self.create_image_array(maskA_path, A=True)
        maskB_images = self.create_image_array(maskB_path)
        testA_images = trainA_images[0]
        testA_images = testA_images[np.newaxis, ...]
        testB_images = trainB_images[0]
        testB_images = testB_images[np.newaxis, ...]

        return {"trainA_images": trainA_images, "trainB_images": trainB_images,
                "testA_images": testA_images, "testB_images": testB_images,
                "maskA_images": maskA_images, "maskB_images": maskB_images}

    @staticmethod
    def create_image_array(image_path, A=False):
        filename = os.listdir(image_path)
        filename.sort()
        img_compressed = np.load(image_path + filename[0])
        if A:
            for i in range(1):
                if i == 0:
                    image = img_compressed['arr_0']
                else:
                    image = np.append(image, img_compressed['arr_{}'.format(i)], axis=0)
                print(i + 1)
                
            #for i in range(40, 50):
            #    image = np.append(image, img_compressed['arr_{}'.format(i)], axis=0)
            #    print(i + 1)
        else:
            image = img_compressed['arr_0']

        print(image.shape)
        print('npz files loaded.')

        return image

    def ck(self, x, k, use_normalization, stride):
        x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1, 1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def conv_block_1(self, i, Base, acti, bn):
        n = Conv2D(filters=Base, kernel_size=3, padding='same')(i)
        n = self.normalization(axis=3, center=True, epsilon=1e-5)(n, training=True) if bn else n
        # n = BatchNormalization()(n) if bn else n
        n = Activation(acti)(n)
        n = Conv2D(filters=Base, kernel_size=3, padding='same')(n)
        n = self.normalization(axis=3, center=True, epsilon=1e-5)(n, training=True) if bn else n
        # n = BatchNormalization()(n) if bn else n
        o = Activation(acti)(n)
        return o

    def conv_block_2(self, i, Base, acti, bn, drop):
        n = MaxPooling2D(pool_size=(2, 2))(i)
        n = Dropout(drop)(n) if drop else n
        o = self.conv_block_1(n, Base, acti, bn)
        return o

    def conv_block_3(self, i, conca_i, Base, acti, bn, drop):
        n = Conv2DTranspose(filters=Base, kernel_size=2, strides=2, padding='same')(i)
        n = concatenate([n, conca_i], axis=3)
        n = Dropout(drop)(n) if drop else n
        o = self.conv_block_1(n, Base, acti, bn)
        return o

    def get_UNet(self, Base, depth, inc_rate, activation, drop, batchnorm, N, name=None):
        i = Input(shape=self.img_shape)
        x_conca = []
        n = self.conv_block_1(i, Base, activation, batchnorm)
        x_conca.append(n)
        for k in range(depth):
            Base = Base * inc_rate
            n = self.conv_block_2(n, Base, activation, batchnorm, drop)
            if k < (depth - 1):
                x_conca.append(n)
        for k in range(depth):
            Base = Base // inc_rate
            n = self.conv_block_3(n, x_conca[-1 - k], Base, activation, batchnorm, drop)

        if N <= 2:
            o = Conv2D(1, (1, 1), activation='sigmoid')(n)
        else:
            o = Conv2D(N, (1, 1), activation='softmax')(n)

        model = Model(inputs=i, outputs=o, name=name)
        return model

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        # Layer 4
        x = self.ck(x, 512, True, 1)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        # x = Activation('sigmoid')(x) - No sigmoid to avoid near-fp32 machine epsilon discriminator cost
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)



        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (
                K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + K.epsilon())

    def dice_coef_loss(self, y_true, y_pred):
        return 1. - self.dice_coef(y_true, y_pred)

        # In[4]:

    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        # if self.channels == 1:
        #     image = image[:, :, 0]

        io.imsave(path_name, image)
        # image = image * 255.
        # im = Image.fromarray(image)
        # im = im.convert("L")
        # im.save(path_name)

    #     toimage(image, cmin=-1, cmax=1).save(path_name)

    def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(os.path.join(directory, 'A')):
            os.makedirs(os.path.join(directory, 'A'))
            os.makedirs(os.path.join(directory, 'B'))
            os.makedirs(os.path.join(directory, 'Atest'))
            os.makedirs(os.path.join(directory, 'Btest'))

        testString = ''

        real_image_Ab = None
        real_image_Ba = None
        for i in range(num_saved_images + 1):
            if i == num_saved_images:
                real_image_A = self.A_test[0]
                real_image_B = self.B_test[0]
                real_image_A = np.expand_dims(real_image_A, axis=0)
                real_image_B = np.expand_dims(real_image_B, axis=0)
                testString = 'test'

            else:
                # real_image_A = self.A_train[rand_A_idx[i]]
                # real_image_B = self.B_train[rand_B_idx[i]]
                if len(real_image_A.shape) < 4:
                    real_image_A = np.expand_dims(real_image_A, axis=0)
                    real_image_B = np.expand_dims(real_image_B, axis=0)

            synthetic_image_B = self.G_A2B.predict(real_image_A)
            synthetic_image_A = self.G_B2A.predict(real_image_B)
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,
                                 'images/{}/{}/epoch{}_sample{}.png'.format(
                                     self.date_time, 'A' + testString, epoch, i))
            self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,
                                 'images/{}/{}/epoch{}_sample{}.png'.format(
                                     self.date_time, 'B' + testString, epoch, i))

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                                 'images/{}/{}.png'.format(
                                     self.date_time, 'tmp'))
        except:  # Ignore if file is open
            pass



    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)

    # ===============================================================================
    # Save and load

    def saveModel(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_models/{}/{}_weights_epoch_{}.h5'.format(self.date_time, model.name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_models/{}/{}_model_epoch_{}.json'.format(self.date_time, model.name, epoch)
        # model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as file:
            # json.dump(json_string, outfile)
            file.write(json_string)

        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):

        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_cycle': self.lambda_cycle,
            'lambda_seg': self.lambda_seg,
            'lambda_d': self.lambda_D,
            'lambda_id': self.lambda_id,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.A_train),
            'number of B train examples': len(self.B_train),
            'number of A test examples': len(self.A_test),
            'number of B test examples': len(self.B_test),
        })

        with open('images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model):
        #         path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
        path_to_weights = os.path.join('generate_images', 'models', '{}.h5'.format(model.name))

        #         model = model_from_json(path_to_model)
        model.load_weights(path_to_weights)

    def load_model_and_generate_synthetic_images(self):
        response = input('Are you sure you want to generate synthetic images instead of training? (y/n): ')[0].lower()
        if response == 'y':
            self.load_model_and_weights(self.G_B2A)
            self.load_model_and_weights(self.G_A2B)
            synthetic_images_A = self.G_B2A.predict(self.B_train)
            reconstructed_images_B = self.G_A2B.predict(synthetic_images_A)

            #np.savez_compressed('./generate_images/2_syn_NCE_imgs.npz',synthetic_images_A)


            def save_image(ori, syn, recon, name, domain):

                image = np.hstack((ori, syn, recon))

                # if self.channels == 1:
                #     image = image[:, :, 0]

                path_ = os.path.join('generate_images', 'synthetic_images', domain, name)
                io.imsave(path_, image)
                # image = image * 255.
                # im = Image.fromarray(image)
                # im = im.convert("L")
                # im.save(os.path.join('generate_images', 'synthetic_images', domain, name))

            #                 image = image * 255.
            #                 Image.fromarray(image, 'L').save()
            #             toimage(image, cmin=-1, cmax=1).save(os.path.join(
            #                 'generate_images', 'synthetic_images', domain, name))

            # Test A images
            for i in range(len(synthetic_images_A)):
                # Get the name from the image it was conditioned on
                name = '{}_synthetic.png'.format(i)
                B_train = self.B_train[i]
                syn_A = synthetic_images_A[i]
                recon_B = reconstructed_images_B[i]
                save_image(B_train, syn_A, recon_B, name, 'hybrid_cv1')

            print('{} synthetic images have been generated and placed in ./generate_images/synthetic_images'
                  .format(len(self.B_train)))


# In[ ]:


class ReflectionPadding2D(Layer):
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'padding': self.padding})
        return config

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


# In[ ]:


if __name__ == '__main__':
    GAN = CycleGAN()
