import parser_args
import tensorflow as tf
import numpy as np
import os
import time
import cv2
from matplotlib import pyplot as plt

import data_management
import regression_network
import utils

SUBDIVISIONS=5
B=3
SHAPE=512

FLAGS = None

def run_test():
    print('Test')
    with tf.device('/cpu:0'):
        keep_prob_pl=tf.placeholder(tf.float32,name='keep_prob_pl')
        path_pl=tf.placeholder(tf.string,name='path_pl')
    
        if FLAGS.path_test_img.endswith('.png'):
            format_img='png'
        elif FLAGS.path_test_img.endswith('.jpg'):
            format_img='jpeg'
        else:
            print('Invalid extension for file %s'%FLAGS.path_test_img)
        
        image=data_manager.load_img_tf(path_pl,format_img)
        image=tf.expand_dims(image,0)
        output=network.inference(image,keep_prob_pl)
        saver=tf.train.Saver()
    
    feed_dict={path_pl:FLAGS.path_test_img,
               keep_prob_pl:1}
    
    with tf.Session() as sess:
        saver.restore(sess,FLAGS.weights)
        print('Weights succesfully loaded')
        tic=time.time()
        out=sess.run(output,feed_dict=feed_dict)
        out=out[0]
        
        print(out[:,:,:,4])
        img=cv2.imread(FLAGS.path_test_img)
        img=output_manager.print_bb(img,out,FLAGS.thresh)
        
                
        print("Predicted in %s s" % (time.time()-tic))
        

        
        cv2.imwrite('prediction.png',img)
    
def run_scoring():
    with tf.device('/cpu:0'):
        with tf.name_scope('Preprocessing'):
            N_files,images_queue=data_manager.def_images_queue(FLAGS.test_pathfile)

    images,labels = images_queue.dequeue_many(1)
    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob_pl')
    output = network.inference(images, keep_prob_pl)

    saver=tf.train.Saver()
    if FLAGS.PRCurve:
        thresholds=np.concatenate([np.linspace(0,0.3,num=30),np.linspace(0.1,1,num=10)])
    else:
        thresholds=np.array([FLAGS.thresh])
    precision=np.zeros((thresholds.shape[0],))
    recall=np.zeros_like(precision)
    
    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess,coord=coord)
        feed_dict={keep_prob_pl:1}
        saver.restore(sess,FLAGS.weights)
        
        for index_thresh,thresh in enumerate(thresholds):
            print("Threshold number %d tried"%(index_thresh+1))
            FP_val=0
            FN_val=0
            TP_val=0
            for i in range(N_files):
                out,lbl=sess.run([output,labels],feed_dict=feed_dict)      
                bb_out=output_manager.list_bb(out[0],thresh,verbose=False,regroup=True)
                bb_lbl=output_manager.list_bb_from_label(lbl[0])
                
                TP_tmp,FP_tmp,FN_tmp=output_manager.premetrics(bb_out,bb_lbl,overlap=0.1)
                TP_val+=TP_tmp
                FP_val+=FP_tmp
                FN_val+=FN_tmp
                print("TP : %d, FP : %d, FN : %d"%(TP_val,FP_val,FN_val),end="\r")
                  
            index_last=index_thresh
            if(TP_val+FP_val==0):
                break
            
            precision[index_thresh]=TP_val/(TP_val+FP_val)
            recall[index_thresh]=TP_val/(TP_val+FN_val)
            
            print()
            print("Precision : %f"%(precision[index_thresh]))
            print("Recall : %f"%(recall[index_thresh]))
        coord.request_stop()
        coord.join()
    
    F_score=2*precision*recall/(precision+recall+1e-6)
    
    if FLAGS.PRCurve:
        plt.figure()
        plt.subplot(211)
        plt.plot(recall[1:index_last],precision[1:index_last])
        plt.title('PR Curve')
        
        plt.subplot(212)
        plt.plot(thresholds[:index_last],F_score[:index_last])
        plt.title('F score depending on the threshold')
        plt.show()
    else:            
        print(F_score[0])
        
def run_training():
    
    if os.path.isdir(FLAGS.log_dir):
        for f in os.listdir(FLAGS.log_dir):
                os.remove('./'+FLAGS.log_dir+'/'+f)
    else :
        os.mkdir(FLAGS.log_dir)
    
    
    with tf.device('/cpu:0'):   
        with tf.name_scope('Preprocessing'):
            N_files,images_queue=data_manager.def_images_queue(FLAGS.training_pathfile)
           
    images,labels = images_queue.dequeue_many(FLAGS.batch_size)
  
    
    

    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob_pl')
    output = network.inference(images, keep_prob_pl)
    loss = network.regression_loss(output, labels)
    lr_pl=tf.placeholder(tf.float32)
    train_op = network.training(loss,lr_pl)
    tf.summary.scalar('Queue fill',images_queue.size(),collections=['losses'])
    merged=tf.summary.merge_all(key='losses')
    saver=tf.train.Saver()
    

    with tf.Session() as sess :# use GPU
        writer=tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
        # start the two queue threads
        if FLAGS.restore:
            saver.restore(sess,FLAGS.weights)
            print('Weights succesfully loaded')
        else:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess,coord=coord)
        #time.sleep(1) # wait for the queue to fill

        for i in range(FLAGS.epochs*N_files//FLAGS.batch_size):
            feed_dict={keep_prob_pl:0.5,
                       lr_pl:FLAGS.learning_rate}
            summary,_, loss_value = sess.run([merged,train_op, loss], feed_dict=feed_dict)
            if i%10==0:
                writer.add_summary(summary,i)
            if i%1000==0:                   
                saver.save(sess,FLAGS.weights)

            print('Iter number :%s, Loss : %s ' %(i,loss_value),end='\r')
        coord.request_stop()
        coord.join()
    
        save_path = saver.save(sess, FLAGS.weights)
        print("Model saved in file: %s" % save_path)           
        
        

if __name__ == '__main__':
    
    parser=parser_args.Parser()
    FLAGS = parser.parse_args()
    
    data_manager=data_management.Data_manager(SHAPE,SUBDIVISIONS)
    network=regression_network.Network(SUBDIVISIONS,B)
    output_manager=utils.Utils(SUBDIVISIONS,SHAPE,B)

    if FLAGS.action=='train':
        run_training()
    elif FLAGS.action=='test':
        run_test()
    elif FLAGS.action=='test_big':
        run_test_big()
    elif FLAGS.action=='score':
        run_scoring()

    else :
        print('Invalid action')
