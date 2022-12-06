# **TensorFlow_GPU2Ascend样例教程**
## 基本信息

**作者：VSRacer**

**创建日期：2022.11.06**

**文档版本：v1.3**

<details>
<table>
<tr><td><b>修改时间</b></td><td><b>文档版本</b></td><td><b>修改内容</b></td><td><b>修改人</b></td></tr>
<tr><td>2022.11.09</td><td>v1.0</td><td>补充工具迁移</td><td>VSRacer</td></tr>
<tr><td>2022.11.10</td><td>v1.1</td><td>补充计算中心上训练</td><td>VSRacer</td></tr>
<tr><td>2022.11.11</td><td>v1.2</td><td>补充引导截图<br>添加训练性能</td><td>VSRacer</td></tr>
<tr><td>2022.11.23</td><td>v1.3</td><td>补充数据集脚本修改<br>手动迁移脚本修改</td><td>VSRacer</td></tr>
</table>
</details>

## 目录
-   [网络模型概述](#网络模型概述)
-   [文件获取](#文件获取)
    -   [镜像准备](#镜像准备)
    -   [模型代码准备](#模型代码准备)
    -   [数据集准备](#数据集准备)
-   [模型迁移](#模型迁移)
    -   [手动迁移](#手动迁移)
    -   [工具迁移](#工具迁移)
-   [训练参数配置](#训练参数配置)
-   [模型训练](#模型训练)
    -   [单Device训练](#单Device训练)
    -   [计算中心上训练](#计算中心上训练)
-   [训练性能](#训练性能)
-   [常见问题FAQ](#常见问题FAQ)

## <h2 id = "网络模型概述">网络模型概述</h2>
-   一个用于场景文本识别的深度神经网络CRNN模型基于TensorFlow的实现。该模型通过CNN网络进行特征提取，之后送入RNN网络并进行CTC损失计算。

-   基于论文：

    ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"]( https://arxiv.org/abs/1507.05717)。

-   源模型参考：

    [Github: MaybeShewill-CV/CRNN_Tensorflow](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)

## <h2 id = "文件获取">文件获取</h2>
### <h3 id = "镜像准备">镜像准备</h3>
-   硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

-   宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

-   本样例演示所使用TensorFlow1.15_arm镜像：[ascend-tensorflow-arm](https://ascendhub.huawei.com/#/detail/ascend-tensorflow-arm)。
### <h3 id = "模型代码准备">模型代码准备</h3>
-   通过Git获取：

        git clone https://github.com/MaybeShewill-CV/CRNN_Tensorflow.git
-   直接下载：

    https://github.com/MaybeShewill-CV/CRNN_Tensorflow/archive/refs/heads/master.zip

### <h3 id = "数据集准备">数据集准备</h3>
-   修改数据集预处理文件
1. 增加头文件`from PIL import Image`注释`import cv2`
2. 修改`/data_provider/tf_io_pipline_fast_tools.py`python文件中`class CrnnFeatureReader(_FeatureIO)`类的读取数据集标签格式：
    ```
    class CrnnFeatureReader(_FeatureIO):
    def __init__(self, char_dict_path, ord_map_dict_path, flags='train'):
        super(CrnnFeatureReader, self).__init__(char_dict_path, ord_map_dict_path)
        self._dataset_flag = flags.lower()
        return

    @property
    def dataset_flags(self):
        return self._dataset_flag

    @dataset_flags.setter
    def dataset_flags(self, value):

        if not isinstance(value, str):
            raise ValueError('Dataset flags shoule be str')

        if value.lower() not in ['train', 'val', 'test']:
            raise ValueError('Dataset flags shoule be within \'train\', \'val\', \'test\'')

        self._dataset_flag = value

    @staticmethod
    def _augment_for_train(input_images, input_labels, input_image_paths,labels_length):
        return input_images, input_labels, input_image_paths,labels_length

    @staticmethod
    def _augment_for_validation(input_images, input_labels, input_image_paths,labels_length):
        return input_images, input_labels, input_image_paths,labels_length

    @staticmethod
    def _normalize(input_images, input_labels, input_image_paths,labels_length):
        input_images = tf.subtract(tf.divide(input_images, 127.5), 1.0)
        return input_images, input_labels, input_image_paths,labels_length

    @staticmethod
    def _extract_features_batch(serialized_batch):
        features = tf.parse_example(
            serialized_batch,
            features={'images': tf.FixedLenFeature([], tf.string),
                      'imagepaths': tf.FixedLenFeature([], tf.string),
                      'labels': tf.VarLenFeature(tf.int64),
                      'labels_length': tf.FixedLenFeature([], tf.int64),
                      }
        )
        bs = features['images'].shape[0]
        images = tf.decode_raw(features['images'], tf.uint8)
        w, h = tuple(CFG.ARCH.INPUT_SIZE)
        images = tf.cast(x=images, dtype=tf.float32)
        images = tf.reshape(images, [bs, h, w, CFG.ARCH.INPUT_CHANNELS])
        
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        label_fixed_shape = np.array([bs, CFG.ARCH.MAX_LENGTH], dtype=np.int32)
        labels = tf.SparseTensor(labels.indices, labels.values, label_fixed_shape)
        labels = tf.sparse_tensor_to_dense(labels, default_value=CFG.ARCH.NUM_CLASSES-1)
        labels_length = features['labels_length']
        imagepaths = features['imagepaths']

        return images, labels, imagepaths,labels_length

    def inputs(self, tfrecords_path, batch_size, num_threads):
        dataset = tf.data.TFRecordDataset(tfrecords_path)

        dataset = dataset.batch(batch_size, drop_remainder=True)

        dataset = dataset.map(map_func=self._extract_features_batch,
                              num_parallel_calls=num_threads)
        if self._dataset_flag == 'train':
            dataset = dataset.map(map_func=self._augment_for_train,
                                  num_parallel_calls=num_threads)
        else:
            dataset = dataset.map(map_func=self._augment_for_validation,
                                  num_parallel_calls=num_threads)
        dataset = dataset.map(map_func=self._normalize,
                              num_parallel_calls=num_threads)

        if self._dataset_flag != 'test':
            dataset = dataset.shuffle(buffer_size=128)
            dataset = dataset.repeat()
        dataset = dataset.prefetch(2) 

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next(name='{:s}_IteratorGetNext'.format(self._dataset_flag))
    ```
3. 修改`/data_provider/tf_io_pipline_fast_tools.py`python文件中，标签修改函数`def sparse_tensor_to_str()`：
    ```
    def sparse_tensor_to_str(self, sparse_tensor):
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        # Translate from consecutive numbering into ord() values
        values_list = []
        for tmp in values:
            if tmp==36:
                values_list.append('1')
            else:
                values_list.append(self._ord_map[str(tmp) + '_index'])
        values = np.array(values_list)

        dense_shape = sparse_tensor.dense_shape

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0],index[1]] = values[i]
        for number_list in number_lists:
            str_lists.append([self.int_to_char(val) for val in number_list])
        for str_list in str_lists:
            res.append(''.join(c for c in str_list if c != '\x00'))
        return res
    ```
4. 修改`/data_provider/tf_io_pipline_fast_tools.py`python文件中，输出tfrecords函数`def _write_tfrecords()`：
    ```
    def _write_tfrecords(tfrecords_writer):
    sess = tf.Session()

    while True:
        sample_info = _SAMPLE_INFO_QUEUE.get()

        if sample_info == _SENTINEL:
            log.info('Process {:d} finished writing work'.format(os.getpid()))
            tfrecords_writer.close()
            break

        sample_path = sample_info[0]
        sample_label = sample_info[1]
        label_length=len(sample_label)
        if _is_valid_jpg_file(sample_path):
            log.error('Image file: {:d} is not a valid jpg file'.format(sample_path))
            continue

        try:
            # try to use PIL 
            image = Image.open(sample_path)
            if image is None:
                continue
            image = image.resize(tuple(CFG.ARCH.INPUT_SIZE),Image.BILINEAR)
            image_np = np.array(image).astype(np.uint8)
            image = image_np.tostring()

        except IOError as err:
            log.error(err)
            continue
        
        features = tf.train.Features(feature={
            'labels': _int64_feature(sample_label),
            'images': _bytes_feature(image),
            'imagepaths': _bytes_feature(sample_path),
            'labels_length':_int64_feature(label_length)

        })
        
        tf_example = tf.train.Example(features=features)
        tfrecords_writer.write(tf_example.SerializeToString())
        log.debug('Process: {:d} get sample from sample_info_queue[current_size={:d}], '
                  'and write it to local file at time: {}'.format(
                   os.getpid(), _SAMPLE_INFO_QUEUE.qsize(), time.strftime('%H:%M:%S')))
    ```
-   本样例使用[Synth 90k](https://www.robots.ox.ac.uk/~vgg/data/text/)数据集为例，下载数据集至代码所在目录：`/CRNN_Tensorflow/data`。在`/data`同级建立目录`/scripts`，存放`prepare_ds.sh`数据预处理脚本，内容如下：
    ```
    CWD=$(cd "$(dirname "$0")"; pwd)
    echo ${CWD}
    cd ${CWD}
    cd ..
    CWD=$(pwd)
    echo ${CWD}
    mkdir -p ${CWD}/data/test
    mkdir -p ${CWD}/data/tfrecords
    # generate tfrecords
    python3 ${CWD}/tools/write_tfrecords.py --dataset_dir=${CWD}/data/mnt/ramdisk/max/90kDICT32px/ \
        --save_dir=${CWD}/data/tfrecords \
        --char_dict_path=${CWD}/data/char_dict/char_dict.json \
        --ord_map_dict_path=${CWD}/data/char_dict/ord_map.json
    ```
-   执行数据集预处理脚本：
    ```
    bash prepare_ds.sh
    ```
## <h2 id = "模型迁移">模型迁移</h2>
### <h3 id = "手动迁移">手动迁移</h3>
已上文提到的github上crnn模型为例进行手动迁移
-   网络结构修改
1. 修改模型使用的数据格式`/CRNN_Tensorflow/crnn_model/crnn_net.py`中的`def __init__()`和`def _init_phase()`函数：
    ```
    def __init__(self, phase, hidden_nums, layers_nums, num_classes):
        super(ShadowNet, self).__init__()

        if phase == 'train':
            self._phase = tf.constant(1, dtype=tf.int8)
        else:
            self._phase = tf.constant(0, dtype=tf.int8)

        self._hidden_nums = hidden_nums
        self._layers_nums = layers_nums
        self._num_classes = num_classes
        self._is_training = self._init_phase()

    def _init_phase(self):
        return tf.equal(self._phase, tf.constant(1, dtype=tf.int8))
    ```
2. 修改模型使用的数据格式`/CRNN_Tensorflow/crnn_model/crnn_net.py`中计算损失的的`def compute_loss()`函数：
    ```
    def compute_loss(self, inputdata, labels, labels_length,name, reuse):
        inference_ret = self.inference(
            inputdata=inputdata, name=name, reuse=reuse
        )
        loss = tf.reduce_mean(
            tf.nn.ctc_loss_v2(
                labels=labels, logits=inference_ret,
                label_length=labels_length,
                logit_length=CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE,dtype=np.int32),
                blank_index=CFG.ARCH.NUM_CLASSES-1
            ),
            name='ctc_loss'
        )
        return inference_ret, loss
    ```
-   修改训练启动`.py`文件，如`/CRNN_Tensorflow/tools/train_shadownet.py`。对python文件增加头文件引用，用于导入基于NPU训练相关文件库：
    ```
    import sys
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    # NPU CONFIGS
    from npu_bridge.estimator import npu_ops
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator.npu.npu_optimizer import allreduce
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    from npu_bridge.hccl import hccl_ops
    ```
-   修改`/CRNN_Tensorflow/config/`下的配置文件`global_config.py`
    ```
    from easydict import EasyDict as edict

    __C = edict()
    # Consumers can get config by: from config import cfg

    cfg = __C

    __C.ARCH = edict()

    # Number of units in each LSTM cell
    __C.ARCH.HIDDEN_UNITS = 256
    # Number of stacked LSTM cells
    __C.ARCH.HIDDEN_LAYERS = 2
    # Sequence length.  This has to be the width of the final feature map of the CNN, which is input size width / 4
    # __C.ARCH.SEQ_LENGTH = 70  # cn dataset
    __C.ARCH.SEQ_LENGTH = 25  # synth90k dataset
    __C.ARCH.MAX_LENGTH = 23  # synth90k dataset
    # Width x height into which training / testing images are resized before feeding into the network
    # __C.ARCH.INPUT_SIZE = (280, 32)  # cn dataset
    __C.ARCH.INPUT_SIZE = (100, 32)  # synth90k dataset
    # Number of channels in images
    __C.ARCH.INPUT_CHANNELS = 3
    # Number character classes
    # __C.ARCH.NUM_CLASSES = 5825  # cn dataset
    __C.ARCH.NUM_CLASSES = 37  # synth90k dataset

    # modified for NPU estimator
    # Save checkpoint every 1000 steps
    __C.SAVE_CHECKPOINT_STEPS=1000
    # Max Checkpoint files
    __C.MAX_TO_KEEP=5
    #data directory
    __C.LOG_DIR="log"
    #
    __C.LOG_NAME="training_log"
    #
    __C.ITERATIONS_PER_LOOP=100

    # Train options
    __C.TRAIN = edict()

    # Use early stopping?
    __C.TRAIN.EARLY_STOPPING = False
    # Wait at least this many epochs without improvement in the cost function
    __C.TRAIN.PATIENCE_EPOCHS = 6
    # Expect at least this improvement in one epoch in order to reset the early stopping counter
    __C.TRAIN.PATIENCE_DELTA = 1e-3

    # Set the shadownet training iterations
    # first choice 
    __C.TRAIN.EPOCHS = 80010

    # Set the display step
    __C.TRAIN.DISPLAY_STEP = 100
    # Set the test display step during training process
    __C.TRAIN.TEST_DISPLAY_STEP = 100
    # Set the momentum parameter of the optimizer
    __C.TRAIN.MOMENTUM = 0.9
    # Set the initial learning rate
    __C.TRAIN.LEARNING_RATE = 0.01
    # Set the GPU resource used during training process
    __C.TRAIN.GPU_MEMORY_FRACTION = 0.9
    # Set the GPU allow growth parameter during tensorflow training process
    __C.TRAIN.TF_ALLOW_GROWTH = True
    # Set the shadownet training batch size
    __C.TRAIN.BATCH_SIZE = 64
    #__C.TRAIN.BATCH_SIZE = 512
    # Set the shadownet validation batch size
    __C.TRAIN.VAL_BATCH_SIZE = 32
    # Set the learning rate decay steps
    __C.TRAIN.LR_DECAY_STEPS = 500000
    # Set the learning rate decay rate
    __C.TRAIN.LR_DECAY_RATE = 0.1
    # Update learning rate in jumps?
    __C.TRAIN.LR_STAIRCASE = True
    # Set multi process nums
    __C.TRAIN.CPU_MULTI_PROCESS_NUMS = 6
    # Set Gpu nums
    __C.TRAIN.GPU_NUM = 2
    # Set moving average decay
    __C.TRAIN.MOVING_AVERAGE_DECAY = 0.9999
    # Set val display step
    __C.TRAIN.VAL_DISPLAY_STEP = 1000

    # Test options
    __C.TEST = edict()

    # Set the GPU resource used during testing process
    __C.TEST.GPU_MEMORY_FRACTION = 0.6
    # Set the GPU allow growth parameter during tensorflow testing process
    __C.TEST.TF_ALLOW_GROWTH = False
    # Set the test batch size
    __C.TEST.BATCH_SIZE = 32
    ```
    根据以下示例对照修改`train_shadownet.py`
    <details>
    <summary style=font-weight:bold>可供参考的代码文件，<code>train_shadownet.py</code>：</summary>
    <pre><code>
    import sys
    import os
    import os.path as ops
    import time
    import math
    import argparse
    
    import tensorflow as tf
    import glog as logger
    import numpy as np
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    
    cur_path = os.path.abspath(os.path.dirname(__file__))
    working_dir = os.path.join(cur_path, '../')
    sys.path.append(working_dir)
    
    from crnn_model import crnn_net
    from local_utils import evaluation_tools
    from config import global_config
    from data_provider import shadownet_data_feed_pipline
    from data_provider import tf_io_pipline_fast_tools
    
    from npu_bridge.estimator import npu_ops
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator.npu.npu_optimizer import allreduce
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    from npu_bridge.hccl import hccl_ops

    CFG = global_config.cfg
    
    def init_args():
        """
        :return: parsed arguments and (updated) config.cfg object
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--root_dir', type=str,default="./",
                            help='Root directory of the project')
        parser.add_argument('-d', '--dataset_dir', type=str,default="data/",
                            help='Directory containing train_features.tfrecords')
        parser.add_argument('-w', '--weights_path', type=str,default=None,
                            help='Path to pre-trained weights to continue training')
        parser.add_argument('-c', '--char_dict_path', type=str,default="data/char_dict/char_dict.json",
                            help='Directory where character dictionaries for the dataset were stored')
        parser.add_argument('-o', '--ord_map_dict_path', type=str,default="data/char_dic/ord_map.json",
                            help='Directory where ord map dictionaries for the dataset were stored')
        parser.add_argument('-s', '--save_dir', type=str,default="./model",
                            help='Directory where checkpoint files will be saved ')
        parser.add_argument('-i', '--num_iters', type=int,default=200000,
                            help='number of training iterations')
        parser.add_argument( '--lr', type=float,default=0.01,
                            help='learning rate per NPU device')
        parser.add_argument('-p', '--lr_sched', type=str,default="cos",
                            help='Directory where checkpoint files will be saved ')
        parser.add_argument( '--momentum', type=float,default=0.9,
                            help='Momentum for sgd optimizer ')
        parser.add_argument('-e', '--decode_outputs', type=args_str2bool, default=False,
                            help='Activate decoding of predictions during training (slow!)')
        parser.add_argument( '--use_nesterov', type=args_str2bool, default=False,
                            help='whether to use nesterov in the sgd optimizer')
        parser.add_argument('-m', '--multi_gpus', type=args_str2bool, default=False,
                            nargs='?', const=True, help='Use multi gpus to train')
        parser.add_argument( '--warmup_step', type=int,default=10,
                            help='number of warmup step used in lr scheduler ')

        #modify for npu overflow
        parser.add_argument("--over_dump", type=str, default="False",
                            help="whether to enable overflow")
        parser.add_argument("--over_dump_path", type=str, default="./",
                            help="path to save overflow dump files")

        return parser.parse_args()
    
    def args_str2bool(arg_value):
        """
        :param arg_value:
        :return:
        """
        if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
    
        elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    def average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)

                grads.append(expanded_g)
    
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
    
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    
        return average_grads
    
    
    def compute_net_gradients(images, labels, net, optimizer=None, is_net_first_initialized=False):
        """
        Calculate gradients for single GPU
        :param images: images for training
        :param labels: labels corresponding to images
        :param net: classification model
        :param optimizer: network optimizer
        :param is_net_first_initialized: if the network is initialized
        :return:
        """
        _, net_loss = net.compute_loss(
            inputdata=images,
            labels=labels,
            name='shadow_net',
            reuse=is_net_first_initialized
        )
    
        if optimizer is not None:
            grads = optimizer.compute_gradients(net_loss)
        else:
            grads = None
    
        return net_loss, grads
    
    def train_shadownet(dataset_dir, weights_path, char_dict_path, ord_map_dict_path,save_dir,args, need_decode=False):
        """
        :param dataset_dir:
        :param weights_path:
        :param char_dict_path:
        :param ord_map_dict_path:
        :param need_decode:
        :return:
        """
        train_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
            dataset_dir=dataset_dir,
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path,
            flags='train'
        )
        val_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
            dataset_dir=dataset_dir,
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path,
            flags='val'
        )

        train_images, train_labels, train_images_paths, train_labels_length = train_dataset.inputs(
            batch_size=CFG.TRAIN.BATCH_SIZE
        )
        
        x, y = np.meshgrid(np.arange(CFG.ARCH.MAX_LENGTH), 
                np.arange(CFG.TRAIN.BATCH_SIZE))
        indexes = np.concatenate([y.flatten()[:, None], x.flatten()[:, None]], axis=1)
        indexes = tf.constant(indexes, dtype=tf.int64)
        train_labels = tf.SparseTensor(indexes, 
                tf.reshape(train_labels, [-1]), 
                np.array([CFG.TRAIN.BATCH_SIZE, CFG.ARCH.MAX_LENGTH], dtype=np.int64))

        val_images, val_labels, val_images_paths,val_labels_length = val_dataset.inputs(
            batch_size=CFG.TRAIN.BATCH_SIZE
        )
        val_labels = tf.SparseTensor(indexes, 
                tf.reshape(val_labels, [-1]), 
                np.array([CFG.TRAIN.BATCH_SIZE, CFG.ARCH.MAX_LENGTH], dtype=np.int64))

        shadownet = crnn_net.ShadowNet(
            phase='train',
            hidden_nums=CFG.ARCH.HIDDEN_UNITS,
            layers_nums=CFG.ARCH.HIDDEN_LAYERS,
            num_classes=CFG.ARCH.NUM_CLASSES
        )

        shadownet_val = crnn_net.ShadowNet(
            phase='test',
            hidden_nums=CFG.ARCH.HIDDEN_UNITS,
            layers_nums=CFG.ARCH.HIDDEN_LAYERS,
            num_classes=CFG.ARCH.NUM_CLASSES
        )

        decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path
        )
    
        train_inference_ret, train_ctc_loss = shadownet.compute_loss(
            inputdata=train_images,
            labels=train_labels,
            labels_length=train_labels_length,
            name='shadow_net',
            reuse=False
        )
        
        val_inference_ret, val_ctc_loss = shadownet_val.compute_loss(
            inputdata=val_images,
            labels=val_labels,
            name='shadow_net',
            labels_length=val_labels_length,
            reuse=True
        )

        train_decoded, train_log_prob = tf.nn.ctc_greedy_decoder(
            train_inference_ret,
            CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
            merge_repeated=False
        )
        val_decoded, val_log_prob = tf.nn.ctc_greedy_decoder(
            val_inference_ret,
            CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
            merge_repeated=False
        )
        
        global_step = tf.train.get_or_create_global_step()
        #rank_size = int(os.getenv('RANK_SIZE'))
        rank_size = int(1)

        warmup_steps = args.warmup_step
        warmup_lr = tf.range(0,args.lr, args.lr/warmup_steps)
        warmup_steps = tf.cast(warmup_steps, tf.int64)
        wp_lr = tf.gather(warmup_lr, tf.minimum(warmup_steps,global_step))
        
        if args.lr_sched=='cos':
    
            decayed_lr = tf.train.cosine_decay(
                learning_rate=args.lr,
                global_step=global_step,
                decay_steps=args.num_iters
            )
        else:
            decayed_lr = tf.train.polynomial_decay(
                learning_rate=args.lr,
                global_step=global_step,
                decay_steps=args.num_iters,
                end_learning_rate=0.000001,
                power=CFG.TRAIN.LR_DECAY_RATE
            )

        learning_rate = tf.cond(
                tf.less(global_step, warmup_steps), 
                lambda:wp_lr,
                lambda: decayed_lr)

        optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=args.momentum,
                use_nesterov=args.use_nesterov)

        optimizer = NPUDistributedOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            opt = optimizer
            gate_gradients = tf.train.Optimizer.GATE_NONE
            grads_and_vars = opt.compute_gradients(train_ctc_loss, gate_gradients=gate_gradients)
            train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

        optimizer = tf.group(train_op)

        tboard_save_dir = save_dir+'/summary'
        os.makedirs(tboard_save_dir, exist_ok=True)
        tf.summary.scalar(name='train_ctc_loss', tensor=train_ctc_loss)
        tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    
        if need_decode:
            train_sequence_dist = tf.reduce_mean(
                tf.edit_distance(tf.cast(train_decoded[0], tf.int32), train_labels),
                name='train_edit_distance'
            )
            val_sequence_dist = tf.reduce_mean(
                tf.edit_distance(tf.cast(val_decoded[0], tf.int32), val_labels),
                name='val_edit_distance'
            )
            tf.summary.scalar(name='train_seq_distance', tensor=train_sequence_dist)
            tf.summary.scalar(name='val_seq_distance', tensor=val_sequence_dist)
    
        merge_summary_op = tf.summary.merge_all()
    
        saver = tf.train.Saver()
        model_save_dir = save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = ops.join(model_save_dir, model_name)
    
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name =  "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["enable_data_pre_proc"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
        custom_op.parameter_map["mix_compile_mode"].b = False

        autotune = False
        autotune = os.environ.get('autotune')
        if autotune:
            autotune = autotune.lower()
            if autotune == 'true':
                print("Autotune module is :" + autotune)
                print("Autotune module has been initiated!")
                custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
            else:
                print("Autotune module is :" + autotune)
                print("Autotune module is enabled or with error setting.")
        else:
            print("Autotune module de_initiate!Pass")

        if args.over_dump == "True":
            print("NPU overflow dump is enabled")
            custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
            custom_op.parameter_map["enable_dump_debug"].b = True
            custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
        else:
            print("NPU overflow dump is disabled")
        
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF 
    
        sess = tf.Session(config=config)
    
        summary_writer = tf.summary.FileWriter(tboard_save_dir)
        summary_writer.add_graph(sess.graph)
    
        train_epochs = args.num_iters
    
        with sess.as_default():
            epoch = 0
            if weights_path is None:
                logger.info('Training from scratch')
                init = tf.global_variables_initializer()
                sess.run(init)
            else:
                logger.info('Restore model from {:s}'.format(weights_path))
                saver.restore(sess=sess, save_path=weights_path)
                epoch = sess.run(tf.train.get_global_step())
            ts_prev = time.time()
            patience_counter = 1
            cost_history = [np.inf]
            while epoch < train_epochs:
                epoch += 1
                if epoch > 1 and CFG.TRAIN.EARLY_STOPPING:
                    if cost_history[-1 - patience_counter] - cost_history[-1] > CFG.TRAIN.PATIENCE_DELTA:
                        patience_counter = 1
                    else:
                        patience_counter += 1
                    if patience_counter > CFG.TRAIN.PATIENCE_EPOCHS:
                        logger.info("Cost didn't improve beyond {:f} for {:d} epochs, stopping early.".
                                    format(CFG.TRAIN.PATIENCE_DELTA, patience_counter))
                        break
    
                if need_decode and epoch % 500 == 0:
                    _, train_ctc_loss_value, train_seq_dist_value, \
                        train_predictions, train_labels_sparse, merge_summary_value = sess.run(
                        [optimizer, train_ctc_loss, train_sequence_dist,
                        train_decoded, train_labels, merge_summary_op])
    
                    train_labels_str = decoder.sparse_tensor_to_str(train_labels_sparse)
                    train_predictions = decoder.sparse_tensor_to_str(train_predictions[0])
                    avg_train_accuracy = evaluation_tools.compute_accuracy(train_labels_str, train_predictions)
    
                    if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                        logger.info('Epoch_Train: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                            epoch + 1, train_ctc_loss_value, train_seq_dist_value, avg_train_accuracy))
    
                    val_ctc_loss_value, val_seq_dist_value, \
                        val_predictions, val_labels_sparse = sess.run(
                        [val_ctc_loss, val_sequence_dist, val_decoded, val_labels])
    
                    val_labels_str = decoder.sparse_tensor_to_str(val_labels_sparse)
                    val_predictions = decoder.sparse_tensor_to_str(val_predictions[0])
                    avg_val_accuracy = evaluation_tools.compute_accuracy(val_labels_str, val_predictions)
    
                    if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                        print('Epoch_Val: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}, time= {}'.format(
                            epoch + 1, val_ctc_loss_value, val_seq_dist_value, avg_val_accuracy, time.time()))
                else:
                    _, train_ctc_loss_value, merge_summary_value,lr_value = sess.run(
                        [optimizer, train_ctc_loss, merge_summary_op,learning_rate])
                        
                    if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                        ts_now = time.time()
                        duration = ts_now - ts_prev
                        step_per_sec = duration / CFG.TRAIN.DISPLAY_STEP
                        fps = (CFG.TRAIN.DISPLAY_STEP * 1.0 / duration ) * CFG.TRAIN.BATCH_SIZE * rank_size
                        ts_prev = ts_now  
                        logger.info('Epoch_Train: {:d} cost= {:9f}, lr= {:9f}, FPS: {:4f}, step_per_sec: {:6f}'.format(epoch , train_ctc_loss_value, lr_value, fps,step_per_sec))
                
                cost_history.append(train_ctc_loss_value)
                summary_writer.add_summary(summary=merge_summary_value, global_step=epoch)
    
                if epoch % 5000 == 0:
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    
            saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
        return np.array(cost_history[1:])
    
    
    def train_shadownet_multi_gpu(dataset_dir, weights_path, char_dict_path, ord_map_dict_path):
        """
    
        :param dataset_dir:
        :param weights_path:
        :param char_dict_path:
        :param ord_map_dict_path:
        :return:
        """
        train_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
            dataset_dir=dataset_dir,
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path,
            flags='train'
        )
        val_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
            dataset_dir=dataset_dir,
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path,
            flags='val'
        )
    
        train_samples = []
        val_samples = []
        for i in range(CFG.TRAIN.GPU_NUM):
            train_samples.append(train_dataset.inputs(batch_size=CFG.TRAIN.BATCH_SIZE))
            val_samples.append(val_dataset.inputs(batch_size=CFG.TRAIN.BATCH_SIZE))
    
        shadownet = crnn_net.ShadowNet(
            phase='train',
            hidden_nums=CFG.ARCH.HIDDEN_UNITS,
            layers_nums=CFG.ARCH.HIDDEN_LAYERS,
            num_classes=CFG.ARCH.NUM_CLASSES
        )
        shadownet_val = crnn_net.ShadowNet(
            phase='test',
            hidden_nums=CFG.ARCH.HIDDEN_UNITS,
            layers_nums=CFG.ARCH.HIDDEN_LAYERS,
            num_classes=CFG.ARCH.NUM_CLASSES
        )
    
        tower_grads = []
        train_tower_loss = []
        val_tower_loss = []
        batchnorm_updates = None
        train_summary_op_updates = None
    
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.polynomial_decay(
            learning_rate=CFG.TRAIN.LEARNING_RATE,
            global_step=global_step,
            decay_steps=CFG.TRAIN.EPOCHS,
            end_learning_rate=0.000001,
            power=CFG.TRAIN.LR_DECAY_RATE
        )
    
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    
        with tf.variable_scope(tf.get_variable_scope()):
            is_network_initialized = False
            for i in range(CFG.TRAIN.GPU_NUM):
                with tf.device('/gpu:{:d}'.format(i)):
                    with tf.name_scope('tower_{:d}'.format(i)) as _:
                        train_images = train_samples[i][0]
                        train_labels = train_samples[i][1]
                        train_loss, grads = compute_net_gradients(
                            train_images, train_labels, shadownet, optimizer,
                            is_net_first_initialized=is_network_initialized)
    
                        is_network_initialized = True
    
                        if i == 0:
                            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                            train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)
    
                        tower_grads.append(grads)
                        train_tower_loss.append(train_loss)
                    with tf.name_scope('validation_{:d}'.format(i)) as _:
                        val_images = val_samples[i][0]
                        val_labels = val_samples[i][1]
                        val_loss, _ = compute_net_gradients(
                            val_images, val_labels, shadownet_val, optimizer,
                            is_net_first_initialized=is_network_initialized)
                        val_tower_loss.append(val_loss)
    
        grads = average_gradients(tower_grads)
        avg_train_loss = tf.reduce_mean(train_tower_loss)
        avg_val_loss = tf.reduce_mean(val_tower_loss)
    
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.TRAIN.MOVING_AVERAGE_DECAY, num_updates=global_step)
        variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
        variables_averages_op = variable_averages.apply(variables_to_average)
    
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)
    
        tboard_save_path = 'tboard/crnn_syn90k_multi_gpu'
        os.makedirs(tboard_save_path, exist_ok=True)
    
        summary_writer = tf.summary.FileWriter(tboard_save_path)
    
        avg_train_loss_scalar = tf.summary.scalar(name='average_train_loss',
                                                tensor=avg_train_loss)
        avg_val_loss_scalar = tf.summary.scalar(name='average_val_loss',
                                                tensor=avg_val_loss)
        learning_rate_scalar = tf.summary.scalar(name='learning_rate_scalar',
                                                tensor=learning_rate)
        train_merge_summary_op = tf.summary.merge(
            [avg_train_loss_scalar, learning_rate_scalar] + train_summary_op_updates
        )
        val_merge_summary_op = tf.summary.merge([avg_val_loss_scalar])
    
        saver = tf.train.Saver()
        model_save_dir = 'model/crnn_syn90k_multi_gpu'
        os.makedirs(model_save_dir, exist_ok=True)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = ops.join(model_save_dir, model_name)
    
        sess_config = tf.ConfigProto(device_count={'GPU': CFG.TRAIN.GPU_NUM}, allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
    
        train_epochs = CFG.TRAIN.EPOCHS
    
        logger.info('Global configuration is as follows:')
        logger.info(CFG)
    
        sess = tf.Session(config=sess_config)
    
        summary_writer.add_graph(sess.graph)
    
        with sess.as_default():
            epoch = 0
            if weights_path is None:
                logger.info('Training from scratch')
                init = tf.global_variables_initializer()
                sess.run(init)
            else:
                logger.info('Restore model from last model checkpoint {:s}'.format(weights_path))
                saver.restore(sess=sess, save_path=weights_path)
                epoch = sess.run(tf.train.get_global_step())
    
            train_cost_time_mean = []
            val_cost_time_mean = []
    
            while epoch < train_epochs:
                epoch += 1
                t_start = time.time()
    
                _, train_loss_value, train_summary, lr = \
                    sess.run(fetches=[train_op,
                                    avg_train_loss,
                                    train_merge_summary_op,
                                    learning_rate])
    
                if math.isnan(train_loss_value):
                    raise ValueError('Train loss is nan')
    
                cost_time = time.time() - t_start
                train_cost_time_mean.append(cost_time)
    
                summary_writer.add_summary(summary=train_summary,
                                        global_step=epoch)
    
                t_start_val = time.time()
    
                val_loss_value, val_summary = \
                    sess.run(fetches=[avg_val_loss,
                                    val_merge_summary_op])
    
                summary_writer.add_summary(val_summary, global_step=epoch)
    
                cost_time_val = time.time() - t_start_val
                val_cost_time_mean.append(cost_time_val)
    
                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} total_loss= {:6f} '
                                'lr= {:6f} mean_cost_time= {:5f}s '.
                                format(epoch + 1,
                                    train_loss_value,
                                    lr,
                                    np.mean(train_cost_time_mean)
                                    ))
                    train_cost_time_mean.clear()
    
                if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                    logger.info('Epoch_Val: {:d} total_loss= {:6f} '
                                ' mean_cost_time= {:5f}s '.
                                format(epoch + 1,
                                    val_loss_value,
                                    np.mean(val_cost_time_mean)))
                    val_cost_time_mean.clear()
    
                if epoch % 5000 == 0:
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
        sess.close()
        return
        
    if __name__ == '__main__':
    
        args = init_args()
    
        if args.multi_gpus:
            logger.info('Use multi gpus to train the model')
            train_shadownet_multi_gpu(
                dataset_dir=args.dataset_dir,
                weights_path=args.weights_path,
                char_dict_path=args.char_dict_path,
                ord_map_dict_path=args.ord_map_dict_path
            )
        else:
            logger.info('Use single gpu to train the model')
            root_dir = args.root_dir
            train_shadownet(
                dataset_dir=os.path.join(root_dir,args.dataset_dir),
                weights_path=args.weights_path,
                char_dict_path=os.path.join(root_dir,args.char_dict_path),
                ord_map_dict_path=os.path.join(root_dir,args.ord_map_dict_path),
                save_dir = args.save_dir,
                args=args,
                need_decode=args.decode_outputs
            )
    </code></pre>
    </details>
### <h3 id = "工具迁移">工具迁移</h3>
-   #### <h4 id = "功能介绍">功能介绍</h4>
       Ascend平台提供了Tensorflow 1.15网络迁移工具，该工具适用于原生的Tensorflow训练脚本迁移场景，AI算法工程师通过该工具分析原生的TensorFlow Python API和Horovod Python API在昇腾AI处理器上的支持度情况，同时将原生的TensorFlow训练脚本自动迁移成昇腾AI处理器支持的脚本。对于无法自动迁移的API，您可以参考工具输出的迁移报告，对训练脚本进行相应的适配修改。
-   #### <h4 id = "获取路径">获取路径</h4>
    a. 已安装CANN软件包  
    -   CANN软件安装完成后，迁移工具在“tfplugin安装目录/tfplugin/latest/python/site-packages/npu_bridge/convert_tf2npu/”目录下。
      
    b. 未安装CANN软件包  
    -   从[昇腾社区](https://gitee.com/ascend/tensorflow/tree/master/convert_tf2npu)获取，直接下载convert_tf2npu文件夹到Linux或Windows环境上任意目录即可。
-   #### <h4 id = "使用限制">使用限制</h4>

    在使用工具进行模型迁移前，先来了解对原始训练脚本的限制：
    
    -   要求原始脚本在GPU/CPU上跑通，精度收敛。
    
    -   要求原始脚本仅使用TensorFlow 1.15官方API和Horovod官方API。若用户脚本使用了其他第三方API，当前工具暂不支持迁移。例如：
    
           1. 不支持原生Keras API，但由于Tensorflow官方API中包括了Tensorflow的Keras API，因此支持Tensorflow的Keras API。
    
           2. 不支持CuPy API，即便原始脚本能在GPU上运行成功，但不能保证在昇腾AI处理器运行成功。
    -   原始脚本中的TensorFlow模块和Horovod模块最好按照如下方式引用，否则工具迁移后，无法生成准确的迁移报告（但并不影响脚本迁移）。
    
           ```
           import tensorflow as tf
           import tensorflow.compat.v1 as tf
           import horovod.tensorflow as hvd
           ```
    
    -   当前版本不支持float64/complex64/complex128/DT_VARIANT数据类型。
    -   当前不支持tf.keras和原生Keras的Loss Scale功能迁移。
    -   当前不支持动态shape网络迁移。
           <details> 
           <summary  style=font-weight:bold>GPU训练复现（可选）</summary>
           在昇腾AI处理器进行模型迁移之前，建议用户事先准备好基于TensorFlow 1.15>开发的训练模型以及配套的数据集，并要求在GPU或CPU上跑通，精度收敛，且达到预期精度和性能要求。同时记录相关精度和性能指标，用于后续在昇腾AI处理器进行精度和性能对比。
           </details>    
-   #### <h4 id = "迁移操作步骤">迁移操作步骤</h4>
    -   安装依赖
    
        ```
        pip3 install pandas
        pip3 install xlrd==1.2.0
        pip3 install openpyxl
        pip3 install tkintertable
        pip3 install google_pasta
        ```
    
    -   训练脚本扫描和自动迁移。
    
        该工具支持在Linux或Windows环境进行脚本迁移。
        -   Linux环境操作步骤
    
            进入迁移工具所在目录，例如“tfplugin安装目录/tfplugin/latest/python/site-packages/npu_bridge/convert_tf2npu/”，执行命令可同时完成脚本扫描和自动迁移，例如：
    
            ```    
            python3 main.py -i /root/models/official/CRNN_Tensorflow
            ```
            参数说明：
            | 参数名 | 必选/可选 | 参数说明  |
            | ----- | --------- | -------- |
            | -i    | 必选      | 被迁移脚本路径文件夹，仅支持单个文件夹     |
            | -o    | 可选      | 迁移后脚本输出路径，不可设置原始脚本子的目录，默认在当前路径下    |
            | -r    | 可选      | 迁移报告输出路径，不可设置原始脚本子的目录，默认在当前路径下    |
            | -m    | 可选      | Python入口文件路径     |
            | -d    | 分布式必选 | 如果原始脚本支持分布式训练，需要指定原始脚本使用的分布式策略，便于工具对分布式脚本进行自动迁移。取值：<br>**tf_strategy**：表示原始脚本使用tf.distribute.Strategy分布式策略<br>**horovod**：表示原始脚本使用horovod分布式策略目前session run分布式脚本无法彻底进行自动迁移，如需支持请参考[Session run脚本支持分布式训练](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha001/moddevg/tfmigr1/atlasmprtgtool_13_0013.html)。|
            | -h    | 可选      | 迁移工具使用帮助。例如：`python3 main.py -h`     |
        -   Windows环境操作步骤

            启动工具：
               
            ```    
            python3 main_win.py
            ```

            在弹出的窗口根据界面提示进行操作：

            1. 根据界面提示设置相关输入。
            
            2. 点击“开始分析”，能看到分析结果，并在指定的路径下生成迁移后的脚本，在指定的“输出分析报告路径”下生成分析报告。

            3. 点击“重新开始分析”，则返回选择路径窗口，可以重新指定输入脚本，再次分析。

            4. 点击“退出”，则退出Tensorflow 1.15网络迁移工具。
    -   迁移过程中，打印如下信息，表明正在扫描相关文件进行脚本迁移，输出日志如下所示：
        ```
        Begin conver file: D:/work/model/CRNN_TensorFlow/crnn_model/crnn_net.py
        Finish conver file: D:/work/model/CRNN_TensorFlow/crnn_model/crnn_net.py

        Begin conver file: D:/work/model/CRNN_TensorFlow/crnn_model/cnn_basenet.py
        Finish conver file: D:/work/model/CRNN_TensorFlow/crnn_model/cnn_basenet.py
        ```
    -   迁移结束后，生成迁移后的脚本，以及迁移报告，输出日志如下所示：
        ```
        Finish conver,output file: D:/Work/20221108/output/; report file: D://Work/20221108/report_npu_20221108161765
        ```
## <h2 id = "训练参数配置">训练参数配置</h2>
-   在整体目录下创建rank table file文件目录configs，例如：`/CRNN_TensorFlow/configs/`，创建`rank_table_8p.json`存放在该目录下，`rank_table_8p.json`内容如下所示：
    ```
    {
        "group_count": "1",
        "group_list": [
        {
            "group_name": "worker",
            "device_count": "8",
            "instance_count": "1",
            "instance_list": [{"devices":
                            [{"device_id":"0","device_ip":"192.168.100.101"},
                                {"device_id":"1","device_ip":"192.168.101.101"},
                                {"device_id":"2","device_ip":"192.168.102.101"},
                                {"device_id":"3","device_ip":"192.168.103.101"},
                                {"device_id":"4","device_ip":"192.168.100.100"},
                                {"device_id":"5","device_ip":"192.168.101.100"},
                                {"device_id":"6","device_ip":"192.168.102.100"},
                                {"device_id":"7","device_ip":"192.168.103.100"}],
                "pod_name":"npu8p",
                "server_id":"127.0.0.1"}]
        }
        ],
        "status": "completed"
    }
    ```
-   为方便对参数信息进行统一管理和调整，编写启动训练的`.bash`文件，存放在和configs同级的test目录下命名为`train_full_1p.sh`，内容如下所示：
    <details>
    <summary  style=font-weight:bold>可供参考的<code>train_full_1p.sh</code>文件</summary>
    <pre><code>cur_path=`pwd`

    export JOB_ID=9999001
    export RANK_SIZE=1
    export RANK_ID=npu8p
    export SLOG_PRINT_TO_STDOUT=0
    export HCCL_CONNECT_TIMEOUT=600
    export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
    RANK_ID_START=0

    data_path=""

    export ASCEND_GLOBAL_LOG_LEVEL=3

    Network="InceptionV4_for_TensorFlow"
    ITERATIONS=240000

    precision_mode="allow_mix_precision"

    over_dump=False
    data_dump_flag=False
    data_dump_step="10"
    profiling=False
    autotune=False

    if [[ $1 == --help || $1 == -h ]];then
        echo"usage:./train_full_1p.sh <args>"
        echo " "
        echo "parameter explain:
        --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
        --over_dump		           if or not over detection, default is False
        --data_dump_flag		   data dump flag, default is 0
        --data_dump_step		   data dump step, default is 10
        --profiling		           if or not profiling for performance debug, default is False
        --autotune                 whether to enable autotune, default is False
        --data_path		           source data of training
        -h/--help		           show help message
        "
        exit 1
    fi

    for para in $*
    do
        if [[ $para == --precision_mode* ]];then
            precision_mode=`echo ${para#*=}`
        elif [[ $para == --over_dump* ]];then
            over_dump=`echo ${para#*=}`
            over_dump_path=${cur_path}/output/overflow_dump
            mkdir -p ${over_dump_path}
        elif [[ $para == --data_dump_flag* ]];then
            data_dump_flag=`echo ${para#*=}`
            data_dump_path=${cur_path}/output/data_dump
            mkdir -p ${data_dump_path}
        elif [[ $para == --data_dump_step* ]];then
            data_dump_step=`echo ${para#*=}`
        elif [[ $para == --profiling* ]];then
            profiling=`echo ${para#*=}`
            profiling_dump_path=${cur_path}/output/profiling
            mkdir -p ${profiling_dump_path}
        elif [[ $para == --autotune* ]];then
            autotune=`echo ${para#*=}`
            export autotune=$autotune
            mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
            mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
            autotune_dump_path=${cur_path}/output/autotune_dump
            mkdir -p ${autotune_dump_path}/GA
            mkdir -p ${autotune_dump_path}/rl
            cp -rf $install_path/fwkacllib/data/tiling/Ascend910/custom ${autotune_dump_path}/GA/
            cp -rf $install_path/fwkacllib/data/rl/Ascend910/custom ${autotune_dump_path}/RL/
        elif [[ $para == --data_path* ]];then
            data_path=`echo ${para#*=}`
        fi
    done

    if [[ $data_path == "" ]];then
        echo "[Error] para \"data_path\" must be confing"
        exit 1
    fi

    start_time=$(date +%s)

    cd $cur_path/../
    for((RANK_ID_n=$RANK_ID_START;RANK_ID_n<$((RANK_SIZE+RANK_ID_START));RANK_ID_n++));
    do
        echo "Device ID: $RANK_ID_n"
        #export RANK_ID_n=$RANK_ID
        export ASCEND_DEVICE_ID=$RANK_ID_n
        ASCEND_DEVICE_ID=$RANK_ID_n
        
        export DEVICE_ID=$RANK_ID_n
        DEVICE_INDEX=$DEVICE_ID
        export DEVICE_INDEX=${DEVICE_INDEX}
        
        if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
            rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
            mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        else
            mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        fi
        
        python3 tools/train_npu.py --dataset_dir=$data_path \
            --char_dict_path=$data_path/char_dict/char_dict.json \
            --ord_map_dict_path=$data_path/char_dict/ord_map.json \
            --save_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt/ \
            --momentum=0.95 \
            --lr=0.08 \
            --use_nesterov=True \
            --warmup_step=8000 \
            --num_iters=${ITERATIONS} \
            --over_dump=${over_dump} \
            --over_dump_path=${over_dump_path} \
            > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    done 
    wait

    end_time=$(date +%s)
    e2e_time=$(( $end_time - $start_time ))

    echo "------------------ Final result ------------------"
    FPS=`grep TimeHistory  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $6}'`
    echo "Final Performance images/sec : $FPS"

    train_accuracy=`grep train_accuracy $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'|cut -c 1-5`
    echo "Final Train Accuracy : ${train_accuracy}"
    echo "E2E Training Duration sec : $e2e_time"

    BatchSize=${batch_size}
    DeviceType=`uname -m`
    CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

    ActualFPS=${FPS}
    TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${RANK_SIZE}'*1000/'${FPS}'}'`

    grep train_loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v BatchTimestamp|awk '{print $10}'|sed 's/,//g'|sed '/^$/d' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

    ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

    echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
    </code></pre>
    </details>
