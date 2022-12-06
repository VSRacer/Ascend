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
