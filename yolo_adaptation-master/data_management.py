import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

N_CHANNELS=1

class Data_manager:
    
    def __init__(self,shape,subdivisions):
        self.shape=shape
        self.subdivisions=subdivisions

    
    def read_bb(self,paths,relative=True) -> np.array:
        """
        create array of labels from <path> It assumes that there is only one object for
        each box
        
        param
            paths: 
                string containing all the labels txt files path
    
        return: numpy array [4*L] of labels
        """
    
        res=np.zeros((len(paths),self.subdivisions,self.subdivisions,5))
        for i,path in enumerate(paths):  
            with open(path.strip(),'r') as f:
                while True :
                    
                    content = f.readline()
                    if content =="":
                        break
                    else:
                        content = content.strip().split()
                        assert(len(content)==5)
                        content = [float(x) for x in content]
                        obj_type=content[0]
                        content=content[1:] 
                        content = np.asarray(content,dtype='float32')
                        bins=[(x+1)/self.subdivisions for x in range(self.subdivisions-1)]
                        obj_i=np.digitize(content[0],bins)
                        obj_j=np.digitize(content[1],bins)
                        content[0]=(content[0]-obj_i/self.subdivisions)*self.subdivisions #relative location in the cell
                        content[1]=(content[1]-obj_j/self.subdivisions)*self.subdivisions
                        res[i,obj_i,obj_j,0]=obj_type+1
                        res[i,obj_i,obj_j,1:]=content
                        
                        
        return(res)
    
    def load_img_tf(self,path,format_img='png'):
        """
        Given a path return the decoded image
        Param:
            path :
                string scalar tensor giving the path
        """
        image_file=tf.read_file(path)
        if format_img=='jpeg':
            image=tf.image.decode_jpeg(image_file,channels=1)
        else:
            image=tf.image.decode_png(image_file,channels=1)
        
        casted=tf.cast(image, tf.float32)/255.
        resized=tf.image.resize_images(images=casted,size=(self.shape,self.shape))
        
    
        
        resized.set_shape([self.shape,self.shape,N_CHANNELS])
        
        mean=tf.reduce_mean(resized)
        std=tf.sqrt(tf.reduce_mean((resized-mean)**2))
        
        final_img=(resized-mean)/std
        
        return(final_img)
                  
    
    def def_images_queue(self,path):
        """
        Function used to create the input pipeline from labelled images:
            - Create one queue containing the individual image paths found in the file located at |path|
            
            - Create one queue containing the images preprocessed
            
            - Link the two queues with the operation to add images to the second queue from the first queue
        """
        f=open(path,'r')
        paths=f.readlines()
        f.close()
        paths=[x.strip() for x in paths]
        if paths[0].endswith('.png'):    
            format_img='png'
            img_extension='.png'
        elif paths[0].endswith('.jpg'):
            format_img='jpeg'
            img_extension='.jpg'
        else:
            print('Invalid format')
            
        lbl_extension='.txt'
    
        paths_lbl=[x.replace(img_extension,lbl_extension) for x in paths]

    
        labels=self.read_bb(paths_lbl)
        # change paths and labels to tf tensors    
        t_paths_train=ops.convert_to_tensor(paths,dtype=tf.string)
        t_labels=ops.convert_to_tensor(labels,dtype=tf.float32)
       
    
        # create queue containing paths and labels joined
        paths_queue=tf.train.slice_input_producer([t_paths_train,t_labels],
                                                  shuffle=True)
        
        # create queue containing loaded and preprocessed images with its label
        # group by batch of FLAGS.batch_size
        images_queue=tf.FIFOQueue(capacity=300,
                                  dtypes=[tf.float32,tf.float32],
                                  shapes=[(self.shape,self.shape,1),(self.subdivisions,self.subdivisions,5)])
        

        
        #operation for filling images_queue
        tuple_to_enqueue=(self.load_img_tf(path=paths_queue[0],format_img=format_img),
                          paths_queue[1])
        enqueue_op=images_queue.enqueue(tuple_to_enqueue)
        N_threads=20
        #create QueueRunner for filling images_queue
        qr=tf.train.QueueRunner(images_queue,[enqueue_op]*N_threads)
        tf.train.add_queue_runner(qr)
        
        return(len(paths),images_queue)
    
    
