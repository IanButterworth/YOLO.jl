
import tensorflow as tf
import math


class Network:
    """Builds the Regression network.

    Implements the tensorflow inference/loss/training pattern for model building.


    """
    def __init__(self,subdivisions,B):

        self.subdivisions=subdivisions
        self.B=B
        self.output_size = self.subdivisions**2*self.B*5
    
    def inference(self,images, keep_prob):
        """
        :param images: Images placeholder and dropout placeholder
        :return: output tensor with coordinates BB
        """
       
    
        with tf.name_scope('hidden_layer_1'):
            weights = tf.Variable(tf.truncated_normal([3, 3, 1, 8],
                                                      stddev=1.0 / math.sqrt(float(3*3))),
                                  name='weights')
            biases = tf.Variable(tf.zeros([8]), name='biases')
            hidden1 = tf.nn.relu(self.conv2d(images, weights) + biases)
            hidden1 = self.maxpool2d(hidden1,4)
    
            activation_map=tf.unstack(hidden1,axis=3)
            for i,am in enumerate(activation_map):
                tf.summary.image('activation map '+str(i+1),tf.expand_dims(am,-1),max_outputs=1)
        with tf.name_scope('hidden_layer_2'):
            weights = tf.Variable(tf.truncated_normal([3, 3, 8, 16],
                                                      stddev=1.0 / math.sqrt(float(3*3*8))),
                                  name='weights')
            biases = tf.Variable(tf.zeros([16]), name='biases')
            hidden2 = tf.nn.relu(self.conv2d(hidden1, weights) + biases)
            hidden2 = self.maxpool2d(hidden2,2)
            
        with tf.name_scope('hidden_layer_3'):
            weights = tf.Variable(tf.truncated_normal([3, 3, 16, 32],
                                                      stddev=1.0 / math.sqrt(float(3*3*16))),
                                  name='weights')
            biases = tf.Variable(tf.zeros([32]), name='biases')
            hidden3 = tf.nn.relu(self.conv2d(hidden2, weights) + biases)
            hidden3 = self.maxpool2d(hidden3,2)
    
        with tf.name_scope('fully_connected'):
            weights = tf.Variable(tf.truncated_normal([32*32*32, 512],
                                                      stddev=1.0 / math.sqrt(float(32*32*32))), 
                                  name='weights')
            biases = tf.Variable(tf.zeros([512]), name='biases')
            fc = tf.reshape(hidden3, [-1, 32 * 32 * 32])
            fc = tf.nn.relu(tf.matmul(fc, weights) + biases)
            fc = tf.nn.dropout(fc, keep_prob)
    
        with tf.name_scope('out'):
            weights = tf.Variable(tf.truncated_normal([512, self.output_size],
                            stddev=1.0 / math.sqrt(float(512))), name='weights')
            biases = tf.Variable(tf.zeros([self.output_size]))
            output = tf.matmul(fc, weights) + biases
            output=tf.sigmoid(output)
            output=tf.reshape(output,[-1,self.subdivisions,self.subdivisions,self.B,5])
    
        return output
    
    
    def ious(self,positions,sizes,positions_lbl,sizes_lbl):
        """
        PARAM:
            positions and sizes : 
                bs*s*s*b*2 
            positions_lbl and sizes_lbl : 
                bs*s*s*2
        
        RETURN:
            bs*s*s*b iou tensor
        """       
        epsilon=1e-15
    
        positions_upper_left_corner_pred=positions-sizes/2*self.subdivisions #bs*s*s*b*2
        positions_lower_right_corner_pred=positions+sizes/2*self.subdivisions #bs*s*s*b*2
        positions_upper_left_corner_lbl=positions_lbl-sizes_lbl/2*self.subdivisions #bs*s*s*2
        positions_lower_right_corner_lbl=positions_lbl+sizes_lbl/2*self.subdivisions #bs*s*s*2
        
        # FOR BROADCASTING  IN THE MAX/MIN OPERATION LATER add one dimension corresponding to b
        positions_upper_left_corner_lbl=tf.expand_dims(positions_upper_left_corner_lbl,axis=3) #bs*s*s*1*2 
        positions_lower_right_corner_lbl=tf.expand_dims(positions_lower_right_corner_lbl,axis=3) #bs*s*s*1*2 
        
        positions_lower_right_corner_intersection=tf.minimum(positions_lower_right_corner_pred,
                                                             positions_lower_right_corner_lbl) #bs*s*s*b*2
        positions_upper_left_corner_intersection=tf.maximum(positions_upper_left_corner_pred,
                                                            positions_upper_left_corner_lbl) #bs*s*s*b*2
        
        intersection_area=tf.reduce_prod(tf.maximum(0.,tf.subtract(positions_lower_right_corner_intersection,
                                                                   positions_upper_left_corner_intersection)),
                                         axis=4,
                                         name='Intersection_area') #bs*s*s*b
                                                             
        union_area=tf.subtract((tf.reduce_prod(sizes,axis=4)+tf.expand_dims(tf.reduce_prod(sizes_lbl,axis=3),axis=3))*self.subdivisions**2,
                               intersection_area,
                               name='Union_area') #bs*s*s*b
        
    
        return_tensor=tf.where(tf.greater(intersection_area,0.),
                               tf.divide(intersection_area+epsilon,union_area+epsilon),
                               tf.zeros_like(intersection_area))
        return(return_tensor)
    
        
        
        
    def regression_loss(self,output, true_labels):
        """
        loss function for training
        
        param:
            output :    
                shape bs*s*s*b*5
            true_labels : 
                shape bs*s*s*5
        returns:
            scalar loss
        """
        
        size_bbox_param=1 # change from yolo : simpler to add another param than take sqrt
        lambda_coord=10
        lambda_noobj=0.5
        
        with tf.name_scope('Cost_function'):
            unstacked=tf.split(output,[2,2,1],axis=4,name='Split_data') # list of two (bs*s*s*b*2) and one bs*s*s*b*1
            unstacked_lbl=tf.split(true_labels,[1,2,2],axis=3,name='Split_label') # list of two (bs*s*s*2) and one bs*s*s*1
            confidence=tf.squeeze(unstacked[2],axis=4) # bs*s*s*b
            boxwise_max=tf.reduce_max(confidence,axis=3) # bs*s*s
            
            cond1=tf.cast(tf.equal(confidence,tf.expand_dims(boxwise_max,axis=-1)),
                          tf.float32,
                          name='Condition_boxj_for_obji') # bs*s*s*b with bs*s*s*1 -> bs*s*s*b
            
            cond2=tf.subtract(1.,
                              tf.cast(tf.equal(unstacked_lbl[0],0),
                                      tf.float32),
                              name='Condition_exists_obji') # bs*s*s*1 with 1 -> bs*s*s*1
            
            with tf.name_scope('Localization_loss'):        
                mse1=tf.reduce_sum(tf.squared_difference(unstacked[0],tf.expand_dims(unstacked_lbl[1],axis=3)),axis=4) #bs*s*s*b*2 and bs*s*s*1*2 -> bs*s*s*b
                mse2=tf.reduce_sum(tf.squared_difference(unstacked[1],tf.expand_dims(unstacked_lbl[2],axis=3)),axis=4) #bs*s*s*b
                tmp1=tf.reduce_mean(tf.reduce_sum(mse1,axis=(1,2,3)))
                tmp2=tf.reduce_mean(unstacked[1])
                tf.summary.scalar('Mse1 bb no cond',tmp1,collections=['losses'])
                tf.summary.scalar('Mse2 bb no cond',tmp2,collections=['losses'])
                
                bbs_loss=tf.reduce_mean(tf.reduce_sum((mse1+size_bbox_param*mse2)*cond1*cond2,axis=(1,2,3)),name='Localization_loss') 
                tf.summary.scalar('Localization_loss',bbs_loss,collections=['losses'])
    
            with tf.name_scope('IOU_loss'):
                
                batch_ious=self.ious(unstacked[0],unstacked[1],unstacked_lbl[1],unstacked_lbl[2]) # bs*s*s*b
                squ_diff=tf.squared_difference(confidence,batch_ious)*cond1 # bs*s*s*b
                ious_loss_obj=tf.reduce_mean(tf.reduce_sum(squ_diff*cond2,axis=(1,2,3)),name='IOU_loss_obj')
                ious_loss_noobj=tf.reduce_mean(tf.reduce_sum(squ_diff*(1-cond2),axis=(1,2,3)),name='IOU_loss_noobj')
    
                tf.summary.scalar('Max_IOU_on_batch',tf.reduce_max(batch_ious),collections=['losses'])
                tf.summary.scalar('IOU_loss_obj',ious_loss_obj,collections=['losses'])
                tf.summary.scalar('IOU_loss_noobj',ious_loss_noobj,collections=['losses']) 
                
                ious_loss=ious_loss_obj+lambda_noobj*ious_loss_noobj
                #tf.summary.scalar('IOU_loss',ious_loss,collections=['losses'])
            
            loss=lambda_coord*bbs_loss+ious_loss
    
            tf.summary.scalar('loss', loss,collections=['losses'])
    
            return loss
    
    
    def training(self,loss,lr):
        """
        Training function
        """
        
        tf.summary.scalar('Learning rate',lr,collections=['losses'])
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        return train_op
    
    
    def conv2d(self,x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    
    
    def maxpool2d(self,x,stride):
        return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
