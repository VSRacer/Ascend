-   #### <h4 id = "开始训练">开始训练</h4>
    a. 使用Notebook环境进行训练
    -   在Notebook中上传数据（**目前仅支持打包上传**）。
    ![在Notebook中上传数据](./img/%E5%9C%A8Notebook%E4%B8%AD%E4%B8%8A%E4%BC%A0%E6%95%B0%E6%8D%AE.png)
    -   从Notebook进入终端。
    ![在Notebook中上传数据](./img/Notebook%E8%BF%9B%E5%85%A5%E7%BB%88%E7%AB%AF.png)
    -   执行训练。
    参考[执行训练](#执行训练)进行网络训练，**训练开始前请进行网络训练参数配置**。
    ![在Notebook中执行训练](./img/Notebook%E6%89%A7%E8%A1%8C%E8%AE%AD%E7%BB%83.png)

    b. 使用ModelArts进行训练
    -   在obs中上传数据，参考[上传数据和脚本](#上传数据和脚本)。
    -   在ModelArts-管理控制台的新版训练任务中创建训练任务。
    ![在ModelArts中创建新版训练任务](./img/ModelArts%E5%88%9B%E5%BB%BA%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1.png)
    -   选择训练镜像和修改代码目录。
    ![在ModelArts中创建新版训练任务](./img/%E5%88%9B%E5%BB%BA%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1.png)
    -   选择训练的资源配置和设置日志输出路径。
    ![选择训练的资源配置和日志输出路径](./img/%E9%85%8D%E7%BD%AE%E8%AE%AD%E7%BB%83%E8%B5%84%E6%BA%90%E5%92%8C%E6%97%A5%E5%BF%97.png)
    -   执行训练。
    参考[执行训练](#执行训练)进行网络训练，**训练开始前请进行网络训练参数配置**。
    ![ModelArts执行训练](./img/ModelArts%E6%89%A7%E8%A1%8C%E8%AE%AD%E7%BB%83.png)

## <h2 id = "训练性能">训练性能</h2>
| **Parameters**             | **Details**                                                     |
| -------------------------- | ----------------------------------------------------------------|
| 环境资源                    | 1*Ascend-910(32GB) ARM: 24核 96GB                               |
| 训练实验时间                | 11/06/2022                                                      |
| TensorFlow版本             | 1.15                                                            |
| 数据集                     | Synth 90k                                                       |
| 训练参数                    |momentum=0.95<br>lr=0.08<br>use_nesterov=True<br>warmup_step=8000|
| Optimizer                  | SGD                                                             |
| Loss Function              | CTCLoss                                                         |
| Loss                       | 0.0025483217                                                    |
| Speed                      | 103ms/step(8pcs)                                                |
| Total time                 | 约 800 mins                                                     |
## <h2 id = "常见问题FAQ">常见问题FAQ</h2>
