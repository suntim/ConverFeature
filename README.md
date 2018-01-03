1/# ConverFeature
def show_rich_feature(x_relu,Node):
    print(x_relu.shape[1],"X",x_relu.shape[2])
    feature_map = tf.reshape(x_relu, [x_relu.shape[1],x_relu.shape[2],Node])
    images = tf.image.convert_image_dtype (feature_map, dtype=tf.uint8)
    images = sess.run(images)
    plt.figure(figsize=(10, 10))#if Node > 25,plot(5,5)
    '''
    for i in np.arange(0, Node):
        plt.subplot(2, 2, i + 1)#you need to change the subplot size if you use other layer
        plt.axis('off')
        plt.imshow(images[:,:,i])
    '''
    for i in np.arange(0, 4):
        plt.subplot(2, 2, i + 1)#you need to change the subplot size if you use other layer
        plt.axis('off')
        plt.imshow(images[:,:,i])
    
    plt.show()


#Run Model
MaxStep = 4
next_batchsize = 1
for j in range(MaxStep):#数据太多，会死机，要分批次训练
    bath1 = mnist.test.next_batch(next_batchsize)#test10000
    test_data = {x:bath1[0],Y_:bath1[1],pkeep:1.0}
    pro,a2,Y1_relu,prv = sess.run([Y,accuracy,Y1,prediction_value],test_data)
    prob1 = pro[0]
    print("第"+str(j)+"批  test的正确率a2为："+str(a2))
    print('test预测的 第 %d 批结果: %s'%(j,prv))
    Label = np.array(np.argmax(bath1[1],1))
    Maxindex = np.array(prv)
    title_name = 'Test Model'
    plot_images(bath1[0], Label,Maxindex,name=title_name)
    fig2=plt.figure('fig2')
    ymin=np.linspace(0,0,n_class)
    for kk in range(len(ymin)):
        h=plt.axvline(kk,ymin[kk],prob1[kk],hold)
        plt.setp(h, 'color', 'r', 'linewidth', 2.0) 
    plt.xlim(-0.5,9.5)
    plt.xlabel('类别')
    plt.ylabel('归属概率')
    title(title_name)
    plt.show(fig2)
    show_rich_feature(Y1_relu,K)

        
    bath2 = mnist.validation.next_batch(next_batchsize)#test10000
    validation_data = {x:bath2[0],Y_:bath2[1],pkeep:1.0}
    pro,a3,Y2_relu,prv = sess.run([Y,accuracy,Y2,prediction_value],validation_data)
    prob1 = pro[0]
    print("第"+str(j)+"批  validation的正确率a3为："+str(a3))
    print('validation预测的 第 %d 批结果: %s'%(j,prv))
    Label = np.array(np.argmax(bath2[1],1))
    Maxindex = np.array(prv)
    title_name = 'Validation Model'
    plot_images(bath2[0], Label,Maxindex,name=title_name)
    fig2=plt.figure('fig2')
    ymin=np.linspace(0,0,n_class)
    for kk in range(len(ymin)):
        h=plt.axvline(kk,ymin[kk],prob1[kk],hold)
        plt.setp(h, 'color', 'r', 'linewidth', 2.0) 
    plt.xlim(-0.5,9.5)
    plt.xlabel('类别')
    plt.ylabel('归属概率')
    title(title_name)
    plt.show(fig2)
    show_rich_feature(Y2_relu,L)

2/# ConverFeature
# conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, Channels, 64])
        biases = bias_variable([64])
        conv1_1 = tf.nn.bias_add(conv2d(x, kernel), biases)
        inputs, pop_mean, pop_var, beta, scale = my_batch_norm(conv1_1)
        conv_batch_norm = tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)
        output_conv1_1 = tf.nn.relu(conv_batch_norm, name=scope)
        # 结果可视化
        split = tf.split(output_conv1_1, num_or_size_splits=64, axis=3)
        tf.summary.image('conv1_1_features', split[0], 64)
