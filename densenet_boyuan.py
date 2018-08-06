import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def block(input_, layers, in_features, new_features_per_layer):
    """
    Simple dense block as in https://arxiv.org/abs/1608.06993 but without batch normalization
    """
    current, features = input_, in_features
    for _ in range(layers):
        new_channels = tf.nn.elu(current)
        new_channels = tf.layers.conv2d(new_channels, filters=new_features_per_layer, kernel_size=3, padding='SAME')
        current = tf.concat((current, new_channels), axis=3)
        features += new_features_per_layer
    return current, features


def upscale(image_, r, color):
    """
    Performs the transformation in Figure 1 of https://arxiv.org/abs/1609.05158
    """
    _, rows, columns, channels = image_.get_shape().as_list()
    transformed_image = image_
    transformed_image = tf.reshape(transformed_image, (-1, rows, columns, r, r, color)) # f, h, w, r, r, c
    transformed_image = tf.transpose(transformed_image, (0, 1, 2, 4, 3, 5))
    transformed_image = tf.split(transformed_image, rows, axis=1) # h, f, 1, w, r, r, c
    transformed_image = tf.concat([tf.squeeze(s, axis=1) for s in transformed_image], axis=2) # f, w, h*r, r, c
    transformed_image = tf.split(transformed_image, columns, axis=1) # w, f, 1, h*r, r, c
    transformed_image = tf.concat([tf.squeeze(s, axis=1) for s in transformed_image], axis=2) # f, h*r, w*r, c
    transformed_image = tf.reshape(transformed_image, (-1, rows * r, columns * r, color)) # f, h*r, w*r, c
    return transformed_image


def run_model(data, dense_blocks, layers_in_block, new_features_per_layer,
              coarsen_factor, learning_rate, training_epochs, minibatch_size=1,color=1):

    coarse_patches_training = data['train_inputs']
    patches_training = data['train_outputs']

    coarse_patches_testing = data['test_inputs']
    patches_testing = data['test_outputs']

    assert patches_training.shape[1] == patches_training.shape[2]
    assert coarse_patches_training.shape[1] == coarse_patches_training.shape[2]

    patch_size = patches_training.shape[1]
    coarse_patch_size = coarse_patches_training.shape[1]

    patch_tf = tf.placeholder(dtype=tf.float32, shape=[None, patch_size, patch_size, color])
    coarse_patch_tf = tf.placeholder(dtype=tf.float32, shape=[None, coarse_patch_size, coarse_patch_size, color])

    current = coarse_patch_tf

    features = current.shape[-1].value
    for _ in range(dense_blocks):
        current, features = block(current, layers_in_block, features, new_features_per_layer)
    current = tf.nn.elu(current)

    current = tf.layers.conv2d(current, filters=coarsen_factor ** 2*color, kernel_size=1, padding='SAME')
    current = tf.nn.relu(current)

    current = upscale(current, patch_size // coarse_patch_size,color=color)

    reconstructed_patch_tf = current

    objective = tf.nn.l2_loss(patch_tf - reconstructed_patch_tf)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(objective)

    objective_mean = tf.nn.l2_loss(tf.reduce_mean(patch_tf,axis=-1) - tf.reduce_mean(reconstructed_patch_tf,axis=-1))

    initialization_operations = tf.global_variables_initializer()

    #saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(initialization_operations)

        def print_objectives(epoch_):

            obj_val_training = sess.run(objective_mean,
                                        feed_dict={coarse_patch_tf: coarse_patches_training[:, :, :, :],
                                                   patch_tf: patches_training[:, :, :, :]})
            obj_val_training /= len(patches_training)

            obj_val_testing = sess.run(objective_mean,
                                       feed_dict={coarse_patch_tf: coarse_patches_testing[:, :, :, :],
                                                  patch_tf: patches_testing[:, :, :, :]})
            obj_val_testing /= len(patches_testing)

            print('epoch {}, log training objective = {}, log testing objective = {}'
                  .format(epoch_, np.log10(obj_val_training), np.log10(obj_val_testing)))
            return [obj_val_training, obj_val_testing]
        a0 = np.array(print_objectives('initialization'))
        lobj = []

        for epoch in range(training_epochs):

            for _ in range(coarse_patches_training.shape[0] // minibatch_size):

                def minibatch_dict():
                    count = coarse_patches_training.shape[0]
                    minibatch = np.random.choice(count, size=minibatch_size, replace=False)
                    return {coarse_patch_tf: coarse_patches_training[minibatch, :, :, :],
                            patch_tf: patches_training[minibatch, :, :, :]}

                sess.run(optimizer, feed_dict=minibatch_dict())

            if epoch % 10==0:
                a = np.array(print_objectives(epoch))
                if abs(a[0]-a0[0])/abs(a0[0])<1e-6:# and a[0]<a0[0]:
                    break
                else:
                    print(abs(a[0]-a0[0])/abs(a0[0]))
                    a0 = a
                    lobj.append(a)
        out_dir = str(coarsen_factor)+'_'+str(dense_blocks)+'_'+str(layers_in_block)+'_'+str(new_features_per_layer)+'_'+str(learning_rate)+'_'+str(color)+'/'
        os.mkdir(out_dir)
        #save_path = saver.save(sess, out_dir+'saved/')
        if color>1:
            lim = []
            lim_coarse = []
        im = None
        im_coarse = None
        for _ in range(5):
            random_patch_index = _ * int(patches_training.shape[0]/4-3)#np.random.choice(len(patches_training))

            coarse_patch = coarse_patches_training[random_patch_index]
            patch = patches_training[random_patch_index]

            feed_dict = {coarse_patch_tf: coarse_patch[np.newaxis, :, :, :]}
            sr = reconstructed_patch_tf.eval(feed_dict=feed_dict)[0]#.squeeze()
            coarse_patch_mean = np.mean(coarse_patch, axis=-1)
            patch_mean = np.mean(patch, axis=-1)
            sr_mean = np.mean(sr, axis=-1)
            if im is None:
                im = np.hstack((patch_mean, sr_mean))
                im_coarse = coarse_patch_mean
            else:
                im = np.vstack((im, np.hstack((patch_mean, sr_mean))))
                im_coarse = np.vstack((im_coarse, coarse_patch_mean))
            if color>1:
                for i in range(color):
                    if _==0:
                        lim.append(np.hstack((patch[:,:,i],sr[:,:,i])))
                        lim_coarse.append(coarse_patch[:,:,i])
                    else:
                        lim[i] = np.vstack((lim[i],np.hstack((patch[:,:,i],sr[:,:,i]))))
                        lim_coarse[i] = np.vstack([lim_coarse[i],coarse_patch[:,:,i]])
        plt.figure()
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(im_coarse)
        plt.suptitle('Gray scale')
        if color==1:
            plt.savefig(out_dir+'comparison_gray_scale.pdf')
        else:
            plt.savefig(out_dir+'comparison_mean.pdf')
        if color>1:
            for i in range(color):
                plt.figure()
                plt.subplot(121)
                plt.imshow(lim[i])
                plt.subplot(122)
                plt.imshow(lim_coarse[i])
                plt.suptitle('Color '+str(i))
                plt.savefig(out_dir+'comparison_'+str(i)+'.pdf')
        #plt.show()
        l2_loss = np.average(np.log10(lobj).T[:,-3:],axis=1)
        return l2_loss


def main(fname, mode = 1, coarsen_factor = 8):

    input_image_file = fname#'Hubble_dust.tif'
    input_image = imageio.imread(input_image_file)
    print('Raw input image shape = {}'.format(input_image.shape))
    if input_image.shape[0]!=input_image.shape[1]:
        _ = min(input_image.shape[0],input_image.shape[1])
        input_image = input_image[:_, :_, :]

    if mode!=0:
        color = input_image.shape[2]
        input_image = input_image[tf.newaxis, :, :, :]
    else:
        grayscale_input_image = np.mean(np.float32(input_image), axis=-1)
        input_image = grayscale_input_image
        color = 1
        input_image = input_image[tf.newaxis, :, :, tf.newaxis]

    with tf.Session():

        input_image = tf.convert_to_tensor(input_image)
        print('input image shape = {}'.format(input_image.shape))

        def is_square(image): return image.shape[1] == image.shape[2]
        assert is_square(input_image)

        target_image_size = 2048

        input_image = tf.image.resize_area(images=input_image,
                                             size=(target_image_size, target_image_size))
        print('reduced input image shape = {}'.format(input_image.shape))

        coarse_input_image = tf.image.resize_area(images=input_image,
                                                    size=(input_image.shape[1].value // coarsen_factor,
                                                          input_image.shape[2].value // coarsen_factor))
        print('coarsened input image shape = {}'.format(coarse_input_image.shape))

        patch_size = 128
        patches = []
        for _ in range(color):
            p = tf.extract_image_patches(images=input_image[:, :, :, _][:, :, :, tf.newaxis],
                                           ksizes=[1, patch_size, patch_size, 1],
                                           strides=[1, patch_size, patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='SAME')
            number_of_patches = p.shape[1].value*p.shape[2].value
            p_ = tf.reshape(p, [number_of_patches, patch_size, patch_size, 1])
            patches.append( p_ )
        patches = tf.concat(patches, axis=3)
        #number_of_patches = patches.shape[0].value
        print('patches shape = {}'.format(patches.shape))

        coarse_patch_size = patch_size // coarsen_factor
        coarse_patches = []
        for _ in range(color):
            cp = tf.extract_image_patches(images=coarse_input_image[:, :, :, _][:, :, :, tf.newaxis],
                                                  ksizes=[1, coarse_patch_size, coarse_patch_size, 1],
                                                  strides=[1, coarse_patch_size, coarse_patch_size, 1],
                                                  rates=[1, 1, 1, 1],
                                                  padding='SAME')
            number_of_coarse_patches = cp.shape[1].value*cp.shape[2].value
            cp_ = tf.reshape(cp, [number_of_coarse_patches, coarse_patch_size, coarse_patch_size, 1])
            coarse_patches.append (cp_ )
        coarse_patches = tf.concat(coarse_patches, axis=3)
        #number_of_coarse_patches = coarse_patches.shape[0].value
        assert number_of_coarse_patches == number_of_patches

        print('decomposed image into {} patches'.format(number_of_patches))

        number_training = np.int(number_of_patches * 0.8)
        print('training with {} patches, testing with {} patches'
              .format(number_training, number_of_patches - number_training))

        patches = patches.eval()#.squeeze()
        coarse_patches = coarse_patches.eval()#.squeeze()

        print('patches have shape {}'.format(patches.shape))
        print('coarse_patches have shape {}'.format(coarse_patches.shape))

    data = {'train_inputs': coarse_patches[:number_training],
            'train_outputs': patches[:number_training],
            'test_inputs': coarse_patches[number_training:],
            'test_outputs': patches[number_training:]}

    out = {}
    """
    lndb = [1,2,3]
    lnl = [2,4,6]
    lnf = [16, 32, 64]
    llr = [3e-3,1e-2,3e-2]
    dndb = []
    for i in lndb:
        dndb.append( run_model(data, dense_blocks=i, layers_in_block=2, new_features_per_layer=32, coarsen_factor=coarsen_factor,
              learning_rate=1e-2, training_epochs = 150*i, minibatch_size=1,color=color) )
    out['ndb'] = np.vstack([[lndb],np.array(dndb).T])
    
    dnl = []
    for i in lnl:
        if i==2:
            dnl.append(dndb[0])
        else:
            dnl.append( run_model(data, dense_blocks=1, layers_in_block=i, new_features_per_layer=32, coarsen_factor=coarsen_factor,
              learning_rate=1e-2, training_epochs = int(150*i/2), minibatch_size=1,color=color) )
    out['nl'] = np.vstack([[lnl],np.array(dnl).T])
    dnf = []
    for i in lnf:
        if i==32:
            dnf.append(dndb[0])
        else:
            dnf.append( run_model(data, dense_blocks=1, layers_in_block=2, new_features_per_layer=i, coarsen_factor=coarsen_factor,
              learning_rate=1e-2, training_epochs = 150, minibatch_size=1,color=color) )
    out['nf'] = np.vstack([[lnf],np.array(dnf).T])
    dlr = []
    for i in llr:
        if i==1e-2:
            dlr.append(dndb[0])
        else:
            dlr.append( run_model(data, dense_blocks=1, layers_in_block=2, new_features_per_layer=32, coarsen_factor=coarsen_factor,
              learning_rate=i, training_epochs = max(150,int(150*1e-2/i)), minibatch_size=1,color=color) )
    out['lr'] = np.vstack([[llr],np.array(dlr).T])
    
    out['l'] = np.vstack([out['ndb'],out['nl'],out['nf'],out['lr']])
    return out
    """
    out = run_model(data, dense_blocks=1, layers_in_block=2, new_features_per_layer=32, coarsen_factor=coarsen_factor,
              learning_rate=1e-2, training_epochs = max(150,int(150*8/coarsen_factor)), minibatch_size=1,color=color)
    return out

def totxt(s, l, ls, t = 0, k = 0):
	j = 0
	with open(s, 'w') as f:
		if t!=0:
			for r in range(len(ls)):
				f.write(ls[r])
				f.write(' ')
			f.write('\n')
		for i in range(len(l[0])):
			if j<k:
				print (l[0][i])
			else:
				for s in range(len(l)):
					f.write(str(l[s][i]))
					f.write(' ')
				f.write('\n')
			j = j+1

def retxt(s, n, k = 1, t = 1):
	out = []
	for i in range(n):
		out.append([])
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				for i in range(n):
					out[i].append(float(lst[i]))
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
	out[i] = np.array(out[i])
	return out

if __name__ == '__main__':
    fname = 'Hubble_Tarantula.jpg'#'slice_box64.png'#
    mode = int(sys.argv[1])

    l = retxt('l2_loss_'+str(mode)+'.txt',12,0,0)
    lcf = [16, 8, 4, 2]
    ld = []
    for i in lcf:
        if i==8:
            ld.append([l[1][0],l[2][0]])
        else:
            ld.append(main(fname,mode,i))
    ld = np.array(ld).T
    totxt('l2_loss_cf_'+str(mode)+'.txt',[lcf,ld[0],ld[1]],0,0,0)
    plt.figure()
    plt.plot(lcf,ld[0],label='Training set')
    plt.plot(lcf,ld[1],'--',label='Test set')
    plt.xlabel('Coarsen factor')
    plt.ylabel(r'$\log(\mathrm{L_{2}\ loss})$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('coarsen_factor.pdf')
    """
    d = main(fname,mode,8)
    totxt('l2_loss_'+str(mode)+'.txt',d['l'],0,0,0)

    l = d['l']
    #l = retxt('l2_loss_'+str(mode)+'.txt',12,0,0)

    plt.figure()
    plt.plot(l[0],l[1],label='Training set')
    plt.plot(l[0],l[2],'--',label='Test set')
    plt.xlabel(r'Number of dense blocks')
    plt.ylabel(r'$\log(\mathrm{L_{2}\ loss})$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('number_of_dense_blocks.pdf')
    
    plt.figure()
    plt.plot(l[3],l[4],label='Training set')
    plt.plot(l[3],l[5],'--',label='Test set')
    plt.xlabel(r'Number of layers per block')
    plt.ylabel(r'$\log(\mathrm{L_{2}\ loss})$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('number_of_layers.pdf')

    plt.figure()
    plt.plot(l[6],l[7],label='Training set')
    plt.plot(l[6],l[8],'--',label='Test set')
    plt.xlabel(r'Number of new features per layer')
    plt.ylabel(r'$\log(\mathrm{L_{2}\ loss})$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('number_of_new_features.pdf')

    plt.figure()
    plt.plot(l[9],l[10],label='Training set')
    plt.plot(l[9],l[11],'--',label='Test set')
    plt.xlabel(r'Learning rate')
    plt.ylabel(r'$\log(\mathrm{L_{2}\ loss})$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_rate.pdf')
    """
    



