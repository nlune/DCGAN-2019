import tensorflow as tf
import numpy as np
import functions as func
import config
import cv2


def get_losses(mask_placeholder, images_placeholder, g_in, g_out, g_loss,l2 = False, rgb = True):
    # returns either l1 or l2 (context + perceptual) inpaint loss, and the gradient
    if rgb:
        axis = [1,2,3]
    else:
        axis = [1,2]

    if not l2:
        context_loss = tf.abs(mask_placeholder * g_out - mask_placeholder * images_placeholder)
        context_loss = tf.reduce_sum(context_loss, axis=axis)
    else:
        context_loss = tf.reduce_sum(tf.square(mask_placeholder * g_out - mask_placeholder * images_placeholder), axis=axis)


    percept_loss = g_loss

    z_loss = context_loss + config.lmda*percept_loss
    z_gradient = tf.gradients(z_loss, g_in)


    return z_loss, z_gradient


def get_best_z_img(masks, images, g_out, g_in, return_bests=False, loss = 1, iters=config.niters):
    # run loss, grad, g_out using masks, g_in, and images
    # backpropagate to input z vec and generate images
    # return image w/ lowest loss & loss value
    # or if return_bests, return the best images & losses of ea. iteration

    z = np.random.randn(BATCH_SIZE, z_dim)
    vel = 0 # initial velocity

    # get loss & gradient
    if loss==2:
      inpaint_loss, gradient = get_losses(masks, images, l2=True)
    else:
      inpaint_loss, gradient = get_losses(masks, images)

    best_img = []
    best_loss = []

    for i in range(iters):
        # feed in batchs of masks & images, and get the inpaint loss, gradient, & output
        feed_dict = {mask_placeholder: masks, images_placeholder: images, g_in: z}
        loss, grad, gen_img = sess.run((inpaint_loss, gradient, g_out), feed_dict=feed_dict)

        grad = grad[0] # because grad.shape is (1,64,100)

        # get the bests of the batch, & append
        best_idx = np.argmin(loss)
        best_img.append(gen_img[best_idx,:,:,:])
        best_loss.append(loss[best_idx])


        v_prev = vel # prev velocity
        vel = v_prev*momentum - r*grad # prev. vel. x momentum - learningrate x gradient
        #z += vel
        z += vel*(1 + momentum) - v_prev*momentum   # dampening momentum

        z = np.clip(z, -1, 1)

    if return_bests:

        return best_img, best_loss
    else:

        return gen_img[best_idx,:,:,:], loss[best_idx]


def get_best_generated(gen_imgs, losses):
    best_z = np.argmin(losses)
    best_gen_img = gen_imgs[best_z,:,:,:]

    return best_gen_img

def inpaint(original_image, generated_image, mask):
    # takes in original image, generated, and mask and returns inpainted image without blending
  boolmsk = np.where(mask > 0)
  invboolmsk = np.where(mask==0)

  inpainted_image = np.zeros(original_image.shape)

  inpainted_image[boolmsk] = original_image[boolmsk]
  inpainted_image[invboolmsk] = generated_image[invboolmsk]


  return inpainted_image
