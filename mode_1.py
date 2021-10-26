import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util
from skimage.measure import compare_ssim as ssim

from os import listdir
import skimage.io
from skimage import data
from skimage import exposure
from skimage.transform import match_histograms
import matplotlib.pyplot as plt

def train(args, model, sess, saver):
    
    if args.fine_tuning :
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for fine-tuning!")
        print("model path is %s"%(args.pre_trained_model))
        
    #num_imgs = len(os.listdir(args.train_Sharp_path))
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs',sess.graph)
    if args.test_with_train:
        f = open("valid_logs.txt", 'w')
    g = open("epoch_log", 'w')
    epoch = 0
    
    
    if args.in_memory:
        
        #blur_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y)
        #sharp_imgs = util.image_loader(args.train_Sharp_path, args.load_X, args.load_Y)
        classes_name = 'oasis'
        ca = os.listdir('/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/'+classes_name)
        #ca.remove(classes_name+'_classes.npy')
        cn = []
        while ca!=[]:
            ci = ca[0]
            ci = ci.split('_')[0]
            ca.remove(ci+'_dataTrain.npz')
            ca.remove(ci+'_labelsTrain.npz')
            ca.remove(ci+'_dataTest.npz')
            ca.remove(ci+'_labelsTest.npz')
            cn.append(ci)
        batch = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/{0}/{1}_dataTrain.npz".format(classes_name,cn[0]))
        batch = batch['payload']
        batch = batch.astype(np.float32)
        for di in range(len(batch)):
            img = batch[di].copy()
            img = img/np.max(img)
            batch[di] = img.copy()
        batch = skimage.color.gray2rgb(batch)
        #batch = batch.reshape((batch.shape[0],batch.shape[1],batch.shape[2],1))

        labels = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/{0}/{1}_labelsTrain.npz".format(classes_name,cn[0]))
        labels = labels['payload']
        labels = labels.astype(np.float32)
        for di in range(len(labels)):
            img = labels[di].copy()
            img = img/np.max(img)
            labels[di] = img.copy()
        labels = skimage.color.gray2rgb(labels)
        #labels = labels.reshape((labels.shape[0],labels.shape[1],labels.shape[2],1))

        batch_all = list(batch)
        label_all = list(labels)



        #matched = match_histograms(batch[min_batch], labels[min_batch], multichannel=True)


        classes_name = 'oasis'
        ca = os.listdir('/home/shared/shangjin_oasis_new/ae_align_DataSets/y_DataSets/'+classes_name)
        #ca.remove(classes_name+'_classes.npy')
        cn = []
        while ca!=[]:
            ci = ca[0]
            ci = ci.split('_')[0]
            ca.remove(ci+'_dataTrain.npz')
            ca.remove(ci+'_labelsTrain.npz')
            ca.remove(ci+'_dataTest.npz')
            ca.remove(ci+'_labelsTest.npz')
            cn.append(ci)
        batch = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/y_DataSets/{0}/{1}_dataTrain.npz".format(classes_name,cn[0]))
        batch = batch['payload']
        batch = batch.astype(np.float32)
        for di in range(len(batch)):
            img = batch[di].copy()
            img = img/np.max(img)
            batch[di] = img.copy()
        batch = skimage.color.gray2rgb(batch)
        #batch = batch.reshape((batch.shape[0],batch.shape[1],batch.shape[2],1))

        labels = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/y_DataSets/{0}/{1}_labelsTrain.npz".format(classes_name,cn[0]))
        labels = labels['payload']
        labels = labels.astype(np.float32)
        for di in range(len(labels)):
            img = labels[di].copy()
            img = img/np.max(img)
            labels[di] = img.copy()
        labels = skimage.color.gray2rgb(labels)
        #labels = labels.reshape((labels.shape[0],labels.shape[1],labels.shape[2],1))

        batch_all.extend(list(batch))
        label_all.extend(list(labels))



        classes_name = 'oasis'
        ca = os.listdir('/home/shared/shangjin_oasis_new/ae_align_DataSets/z_DataSets/'+classes_name)
        #ca.remove(classes_name+'_classes.npy')
        cn = []
        while ca!=[]:
            ci = ca[0]
            ci = ci.split('_')[0]
            ca.remove(ci+'_dataTrain.npz')
            ca.remove(ci+'_labelsTrain.npz')
            ca.remove(ci+'_dataTest.npz')
            ca.remove(ci+'_labelsTest.npz')
            cn.append(ci)
        batch = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/z_DataSets/{0}/{1}_dataTrain.npz".format(classes_name,cn[0]))
        batch = batch['payload']
        batch = batch.astype(np.float32)
        for di in range(len(batch)):
            img = batch[di].copy()
            img = img/np.max(img)
            batch[di] = img.copy()
        batch = skimage.color.gray2rgb(batch)
        #batch = batch.reshape((batch.shape[0],batch.shape[1],batch.shape[2],1))

        labels = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/z_DataSets/{0}/{1}_labelsTrain.npz".format(classes_name,cn[0]))
        labels = labels['payload']
        labels = labels.astype(np.float32)
        for di in range(len(labels)):
            img = labels[di].copy()
            img = img/np.max(img)
            labels[di] = img.copy()
        labels = skimage.color.gray2rgb(labels)
        #labels = labels.reshape((labels.shape[0],labels.shape[1],labels.shape[2],1))

        batch_all.extend(list(batch))
        label_all.extend(list(labels))

        batch = np.array(batch_all)
        labels = np.array(label_all)
        


        blur_imgs =  batch * 255.#none,256,256,3
        sharp_imgs = labels * 255.#none,256,256,3
        num_imgs = batch.shape[0]
        step = num_imgs // args.batch_size

        tar_batch = np.array([sharp_imgs[0] for jj in range(args.batch_size)])


        
        while epoch < args.max_epoch:
            random_index = np.random.permutation(len(blur_imgs))
            for k in range(step):
                s_time = time.time()
                blur_batch, sharp_batch = util.batch_gen(blur_imgs, sharp_imgs, args.patch_size, args.batch_size, random_index, k, args.augmentation)
                
                ########
                #blur_batch1 = skimage.color.gray2rgb(blur_batch)
                #sharp_batch1 = skimage.color.gray2rgb(sharp_batch)


                blur_batch = match_histograms(blur_batch, tar_batch, multichannel=True)


                sharp_batch = match_histograms(sharp_batch, tar_batch, multichannel=True)





                #blur_batch = skimage.color.rgb2gray(blur_batch)
                #sharp_batch = skimage.color.rgb2gray(sharp_batch)
                #blur_batch = blur_batch.reshape((blur_batch.shape[0],blur_batch.shape[1],blur_batch.shape[2],1))
                #sharp_batch = sharp_batch.reshape((sharp_batch.shape[0],sharp_batch.shape[1],sharp_batch.shape[2],1))

                ########
                for t in range(args.critic_updates):
                    _, D_loss = sess.run([model.D_train, model.D_loss], feed_dict = {model.blur : blur_batch, model.sharp : sharp_batch, model.epoch : epoch})
                    
                _, G_loss, d_loss_real, d_loss_fake, content_loss, GP_loss, mse_loss = sess.run([model.G_train, model.G_loss, model.d_loss_real, model.d_loss_fake, model.content_loss, model.GP_loss, model.mse_loss], feed_dict = {model.blur : blur_batch, model.sharp : sharp_batch, model.epoch : epoch})
                
                g.write('epoch ' + str(epoch)+ 'batch ' + str(k) + ' --- D_loss : ' + str(D_loss) + ' --- G_loss : ' + str(G_loss)+ ' --- d_loss_real : ' + str(d_loss_real)+ ' --- d_loss_fake : ' + str(d_loss_fake) + ' --- content_loss : ' + str(content_loss) + ' --- GP_loss : ' + str(GP_loss) + ' --- mse_loss : ' + str(mse_loss) + '\n')
                g.flush()

                e_time = time.time()
            
            if epoch % args.log_freq == 0:
                summary = sess.run(merged, feed_dict = {model.blur : blur_batch, model.sharp: sharp_batch})
                train_writer.add_summary(summary, epoch)
                if args.test_with_train:
                    test(args, model, sess, saver, f, epoch, loading = False)

                #print("%d training epoch completed" % epoch)
                #print("D_loss : %0.4f, \t G_loss : %0.4f"%(D_loss, G_loss))
                #print("Elpased time : %0.4f"%(e_time - s_time))
                

            if ((epoch) % args.model_save_freq ==0):
                saver.save(sess, './model/DeblurrGAN', global_step = epoch, write_meta_graph = False)
            
            epoch += 1

        saver.save(sess, './model/DeblurrGAN_last', write_meta_graph = False)
    
    else:
        while epoch < args.max_epoch:
            
            sess.run(model.data_loader.init_op['tr_init'])
            
            for k in range(step):
                s_time = time.time()
                
                for t in range(args.critic_updates):
                    _, D_loss = sess.run([model.D_train, model.D_loss], feed_dict = {model.epoch : epoch})
                    
                _, G_loss = sess.run([model.G_train, model.G_loss], feed_dict = {model.epoch : epoch})
                             
                e_time = time.time()
            
            if epoch % args.log_freq == 0:
                summary = sess.run(merged)
                train_writer.add_summary(summary, epoch)
                if args.test_with_train:
                    test(args, model, sess, saver, f, epoch, loading = False)
                print("%d training epoch completed" % epoch)
                print("D_loss : %0.4f, \t G_loss : %0.4f"%(D_loss, G_loss))
                print("Elpased time : %0.4f"%(e_time - s_time))
            if ((epoch) % args.model_save_freq ==0):
                saver.save(sess, './model/DeblurrGAN', global_step = epoch, write_meta_graph = False)
            
            epoch += 1

        saver.save(sess, './model/DeblurrGAN_last', global_step = epoch, write_meta_graph = False)
        
    if args.test_with_train:
        f.close()
        
        
def test(args, model, sess, saver, file, step = -1, loading = False):
        
    if loading:
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for test!")
        print("model path is %s"%args.pre_trained_model)
        
    blur_img_name = sorted(os.listdir(args.test_Blur_path))
    sharp_img_name = sorted(os.listdir(args.test_Sharp_path))
    
    PSNR_list = []
    ssim_list = []
    
    if args.in_memory :
        
        blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train = False)
        sharp_imgs = util.image_loader(args.test_Sharp_path, args.load_X, args.load_Y, is_train = False)
        
        for i, ele in enumerate(blur_imgs):
            blur = np.expand_dims(ele, axis = 0)
            sharp = np.expand_dims(sharp_imgs[i], axis = 0)
            output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim], feed_dict = {model.blur : blur, model.sharp : sharp})
            if args.save_test_result:
                output = Image.fromarray(output[0])
                split_name = blur_img_name[i].split('.')
                output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))

            PSNR_list.append(psnr)
            ssim_list.append(ssim)

    else:
        
        sess.run(model.data_loader.init_op['val_init'])

        for i in range(len(blur_img_name)):
            
            output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim])
            
            if args.save_test_result:
                output = Image.fromarray(output[0])
                split_name = blur_img_name[i].split('.')
                output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))
                
            PSNR_list.append(psnr)
            ssim_list.append(ssim)
            
    length = len(PSNR_list)
    
    mean_PSNR = sum(PSNR_list) / length
    mean_ssim = sum(ssim_list) / length
    
    if step == -1:
        file.write('PSNR : 0.4f SSIM : %0.4f'%(mean_PSNR, mean_ssim))
        file.close()
        
    else :
        file.write("%d-epoch step PSNR : %0.4f SSIM : %0.4f \n"%(step, mean_PSNR, mean_ssim))


            
def test_only(args, model, sess, saver):
    
    saver.restore(sess,args.pre_trained_model)
    print("saved model is loaded for test only!")
    print("model path is %s"%args.pre_trained_model)
    
    #blur_img_name = sorted(os.listdir(args.test_Blur_path))

    if args.in_memory :
        classes_name = 'oasis'
        ca = os.listdir('/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/'+classes_name)
        #ca.remove(classes_name+'_classes.npy')
        cn = []
        while ca!=[]:
            ci = ca[0]
            ci = ci.split('_')[0]
            ca.remove(ci+'_dataTrain.npz')
            ca.remove(ci+'_labelsTrain.npz')
            ca.remove(ci+'_dataTest.npz')
            ca.remove(ci+'_labelsTest.npz')
            cn.append(ci)

        labels = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/{0}/{1}_labelsTrain.npz".format(classes_name,cn[0]))
        labels = labels['payload']
        labels = labels.astype(np.float32)
        for di in range(len(labels)):
            img = labels[di].copy()
            img = img/np.max(img)
            labels[di] = img.copy()
        labels = skimage.color.gray2rgb(labels)

        labels=labels*255.

        
        tar_batch = np.array([labels[0] for jj in range(args.batch_size)])


        classes_name = 'oasis'
        ca = os.listdir('/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/'+classes_name)
        #ca.remove(classes_name+'_classes.npy')
        cn = []
        while ca!=[]:
            ci = ca[0]
            ci = ci.split('_')[0]
            ca.remove(ci+'_dataTrain.npz')
            ca.remove(ci+'_labelsTrain.npz')
            ca.remove(ci+'_dataTest.npz')
            ca.remove(ci+'_labelsTest.npz')
            cn.append(ci)
        batch = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/{0}/{1}_dataTest.npz".format(classes_name,cn[0]))
        batch = batch['payload']
        batch = batch.astype(np.float32)
        for di in range(len(batch)):
            img = batch[di].copy()
            img = img/np.max(img)
            batch[di] = img.copy()
        batch = skimage.color.gray2rgb(batch)
        #batch = batch.reshape((batch.shape[0],batch.shape[1],batch.shape[2],1))

        labels = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets/{0}/{1}_labelsTest.npz".format(classes_name,cn[0]))
        labels = labels['payload']
        labels = labels.astype(np.float32)
        for di in range(len(labels)):
            img = labels[di].copy()
            img = img/np.max(img)
            labels[di] = img.copy()
        labels = skimage.color.gray2rgb(labels)
        #labels = labels.reshape((labels.shape[0],labels.shape[1],labels.shape[2],1))

        batch_all = list(batch)
        label_all = list(labels)


        classes_name = 'oasis'
        ca = os.listdir('/home/shared/shangjin_oasis_new/ae_align_DataSets/y_DataSets/'+classes_name)
        #ca.remove(classes_name+'_classes.npy')
        cn = []
        while ca!=[]:
            ci = ca[0]
            ci = ci.split('_')[0]
            ca.remove(ci+'_dataTrain.npz')
            ca.remove(ci+'_labelsTrain.npz')
            ca.remove(ci+'_dataTest.npz')
            ca.remove(ci+'_labelsTest.npz')
            cn.append(ci)
        batch = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/y_DataSets/{0}/{1}_dataTest.npz".format(classes_name,cn[0]))
        batch = batch['payload']
        batch = batch.astype(np.float32)
        for di in range(len(batch)):
            img = batch[di].copy()
            img = img/np.max(img)
            batch[di] = img.copy()
        batch = skimage.color.gray2rgb(batch)
        #batch = batch.reshape((batch.shape[0],batch.shape[1],batch.shape[2],1))

        labels = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/y_DataSets/{0}/{1}_labelsTest.npz".format(classes_name,cn[0]))
        labels = labels['payload']
        labels = labels.astype(np.float32)
        for di in range(len(labels)):
            img = labels[di].copy()
            img = img/np.max(img)
            labels[di] = img.copy()
        labels = skimage.color.gray2rgb(labels)

        batch_all.extend(list(batch))
        label_all.extend(list(labels))



        classes_name = 'oasis'
        ca = os.listdir('/home/shared/shangjin_oasis_new/ae_align_DataSets/z_DataSets/'+classes_name)
        #ca.remove(classes_name+'_classes.npy')
        cn = []
        while ca!=[]:
            ci = ca[0]
            ci = ci.split('_')[0]
            ca.remove(ci+'_dataTrain.npz')
            ca.remove(ci+'_labelsTrain.npz')
            ca.remove(ci+'_dataTest.npz')
            ca.remove(ci+'_labelsTest.npz')
            cn.append(ci)
        batch = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/z_DataSets/{0}/{1}_dataTest.npz".format(classes_name,cn[0]))
        batch = batch['payload']
        batch = batch.astype(np.float32)
        for di in range(len(batch)):
            img = batch[di].copy()
            img = img/np.max(img)
            batch[di] = img.copy()
        batch = skimage.color.gray2rgb(batch)
        #batch = batch.reshape((batch.shape[0],batch.shape[1],batch.shape[2],1))

        labels = np.load("/home/shared/shangjin_oasis_new/ae_align_DataSets/z_DataSets/{0}/{1}_labelsTest.npz".format(classes_name,cn[0]))
        labels = labels['payload']
        labels = labels.astype(np.float32)
        for di in range(len(labels)):
            img = labels[di].copy()
            img = img/np.max(img)
            labels[di] = img.copy()
        labels = skimage.color.gray2rgb(labels)


        batch_all.extend(list(batch))
        label_all.extend(list(labels))

        batch = np.array(batch_all)
        labels = np.array(label_all)






        #matched = match_histograms(batch[min_batch], labels[min_batch], multichannel=True)

        blur_imgs =  batch * 255.#none,256,256,3
        sharp_imgs = labels * 255.#none,256,256,3
        num_imgs = batch.shape[0]
        step = num_imgs // args.batch_size

        for k in range(step):
            random_index = range(len(blur_imgs))
            blur_batch, sharp_batch = util.batch_gen(blur_imgs, sharp_imgs, args.patch_size, args.batch_size, random_index, k, args.augmentation)
                
            ########

            blur_batch = match_histograms(blur_batch, tar_batch, multichannel=True)

            sharp_batch = match_histograms(sharp_batch, tar_batch, multichannel=True)

            #blur_batch = match_histograms(blur_batch, sharp_batch, multichannel=True)
            #blur_batch = skimage.color.rgb2gray(blur_batch)
            #sharp_batch = skimage.color.rgb2gray(sharp_batch)
            #blur_batch = blur_batch.reshape((blur_batch.shape[0],blur_batch.shape[1],blur_batch.shape[2],1))
            #sharp_batch = sharp_batch.reshape((sharp_batch.shape[0],sharp_batch.shape[1],sharp_batch.shape[2],1))
            


            output = sess.run(model.output, feed_dict = {model.blur : blur_batch})

            if k<3 or (k>(step//3) and k<(step//3+4)) or (k>(2*step//3) and k<(2*step//3+4)):
                #for kk in range(args.batch_size):
                for kk in range(8):
                    image3=[]
                    image3.append(blur_batch[kk])
                    image3.append(sharp_batch[kk])
                    image3.append(output[kk])
                    for ikk in range(3):
                        plt.figure()
                        #image3[ikk] = image3[ikk].astype(np.float32)
                        #image3[ikk] = skimage.color.gray2rgb(image3[ikk])
                        image3[ikk] = image3[ikk].astype(np.uint8)
                        image3[ikk] = image3[ikk].reshape(image3[ikk].shape[0],image3[ikk].shape[1],3)
                        #image3[ikk] = skimage.color.gray2rgb(image3[ikk])
                        skimage.io.imshow(image3[ikk])
                        plt.savefig('./images/pic'+str(k*args.batch_size+kk)+'_'+str(ikk)+'.png')




        
        #blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train = False)

        '''
        
        for i, ele in enumerate(blur_imgs):
            blur = np.expand_dims(ele, axis = 0)
            
            if args.chop_forward:
                output = util.recursive_forwarding(blur, args.chop_size, sess, model, args.chop_shave)
                output = Image.fromarray(output[0])
            
            else:
                output = sess.run(model.output, feed_dict = {model.blur : blur})
                output = Image.fromarray(output[0])
            
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))
        '''

    else:
        
        sess.run(model.data_loader.init_op['te_init'])

        for i in range(len(blur_img_name)):
            output = sess.run(model.output)
            output = Image.fromarray(output[0])
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))    

