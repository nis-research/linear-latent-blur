# def build_generator(self, name=None):
#     def residual_block(layer_input, filters):
#         """Residual block described in paper"""
#         d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
#         d = InstanceNormalization()(d)
#         d = Activation('relu')(d)
#         d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
#         d = InstanceNormalization()(d)
#         d = Add()([d, layer_input])
#         return d
#
#     # Low resolution image input
#     img_lr = Input(shape=self.hr_shape)
#
#     # with tf.device('/gpu:0') :
#     # Pre-residual block
#     # c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
#     c1 = Conv2D(64, kernel_size=7, strides=1, padding='same')(img_lr)
#     c1 = InstanceNormalization()(c1)
#     c1 = Activation('relu')(c1)
#
#     n_downsampling = 2
#     for i in range(n_downsampling):
#         mult = 2 ** i
#         c1 = Conv2D(filters=64 * mult * 2, kernel_size=(3, 3), strides=2, padding='same')(c1)
#         c1 = InstanceNormalization()(c1)
#         c1 = Activation('relu')(c1)
#
#     # Propogate through residual blocks
#     r = residual_block(c1, self.gf * (n_downsampling ** 2))
#     for _ in range(8):
#         r = residual_block(r, self.gf * (n_downsampling ** 2))
#
#     for i in range(n_downsampling):
#         mult = 2 ** (n_downsampling - i)
#         r = UpSampling2D()(r)
#         r = Conv2D(filters=int(64 * mult / 2), kernel_size=(3, 3), padding='same')(r)
#         r = InstanceNormalization()(r)
#         r = Activation('relu')(r)
#
#         # Post-residual block
#     c2 = Conv2D(self.channels, kernel_size=7, strides=1, padding='same')(r)
#     c2 = Activation('tanh')(c2)
#     c2 = Add()([c2, img_lr])
#     model = Model(img_lr, [c2], name=name)
#
#     model.summary()
#     return model