import siot
import time
import binascii
from fastapi import FastAPI, UploadFile, File
import numpy as np
import shutil
import uvicorn
import socket
import scipy.io.wavfile as wav
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from BaseDT.BaseData import ImageData
import onnxruntime as rt
import wave


# 1. 消息通信协议
# 1.1 SIoT通信方式

def siot_init(CLIENT_ID, SERVER, IOT_UserName, IOT_PassWord):
    """
    功能：siot连接初始化，与siot通信上位机进行连接
    :param CLIENT_ID: 在SIoT上，CLIENT_ID可以留空，但在实际使用中建议设为非空
    :param SERVER: MQTT服务器IP
    :param IOT_UserName: 用户名
    :param IOT_PassWord: 密码
    :return:
    """
    siot.init(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
    siot.connect()
    siot.loop()
    return siot


def publish_msg(IOT_pubTopic, msg, step_size=None):
    """
    功能：向siot上位机发送报文
    :param IOT_pubTopic: 设置发送主题
    :param msg: 需要发送的消息
    :param step_size: 设置是否需要分片传输，默认为整报文传送。面向资源极度受限的板子（售价约几十块），
                        可能需要通过报文分片传输的方式进行与上位机的消息通信，避免内存超限
    :return:
    """
    if step_size is not None:
        # todo: check step send file by tag
        for index in range(0, len(msg), step_size):
            b = binascii.b2a_base64(msg[index:min(index + step_size, len(msg))])
            siot.client.publish(IOT_pubTopic, b)
            time.sleep_ms(10)
    else:
        siot.client.publish(IOT_pubTopic, msg)


def sub_relay(client, userdata, msg):
    """
    功能：监听siot消息通信的信息
    :param client: 设置接收主题
    :param userdata:
    :param msg: 接收到的消息面板，需要进行解码
    :return:
    """
    global message_tol
    global step_size
    payload = msg.payload.decode()
    receive_msg = binascii.a2b_base64(payload)
    if step_size is not None:
        if isinstance(receive_msg, bytes) and len(receive_msg) > 0:
            message_tol += receive_msg
            if len(receive_msg) != step_size:
                # goto infer
                pass
        else:
            message_tol = b''


def receive_msg(IOT_recTopic):
    """
    :param IOT_recTopic: 设置监听消息的主题
    :return:
    """
    siot.client.subscribe(IOT_recTopic, sub_relay)


# 1.2 EasyAPI通信方式

app = FastAPI()
label = []


@app.post("/upload")
async def upload_file(files: UploadFile = File(...)):
    fileUpload = f"./{files.filename}"
    with open(fileUpload, "wb") as buffer:
        shutil.copyfileobj(files.file, buffer)
    input_data = wav2tensor(fileUpload, 'MobileNet')
    ckpt_path = ''
    ort_output = infer_onnx(ckpt_path, input_data)
    idx = np.argmax(ort_output, axis=1)[0]
    str_pub = '结果：' + label[idx] + ' 置信度：' + str(round(ort_output[0][idx], 2))
    return str_pub


def get_host_ip():
    """
    返回本地IP地址
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def easyapi_init(port):
    """
    上位机在启动easyapi时会自动发布接口地址，供其他设备调用
    :param port: 设置想要开放的端口号
    :return:
    """
    ip = str(get_host_ip())
    print("接口地址为http://" + ip + "/upload")
    uvicorn.run(app=app, host="0.0.0.0", port=port, workers=1)


# 2. 文件预处理函数

def wav2tensor(file_path, backbone):
    """
    对wav音频文件进行数据预处理，返回推理机需要的数据类型
    :param file_path: wav音频文件地址
    :param backbone: 推理机网络模型选择
    :return:推理机需要的数据类型
    """
    viridis_cmap = plt.get_cmap('viridis')
    color_map = viridis_cmap.colors
    color_map = (np.array(color_map) * 255).astype(np.uint8)
    fs = 16000
    if isinstance(file_path, str):
        fs0, wave = wav.read(file_path)  # 读取原始采样频率和数据
        if fs0 != fs:  # 如果采样频率不是16000则重采样
            num = int(len(wave) * fs / fs0)  # 计算目标采样点数
            wave = signal.resample(wave, num)  # 对数据进行重采样
    spec = librosa.feature.melspectrogram(wave, sr=fs, n_fft=512)
    spec = librosa.power_to_db(spec, ref=np.max)

    spec_new = (((spec + 80) / 80) * 255).astype(np.uint8)
    h, w = spec_new.shape
    rgb_matrix = np.array([color_map[i] for i in spec_new.flatten()]).reshape(h, w, 3)

    image = Image.fromarray(rgb_matrix.astype(np.uint8))
    image = np.array(image)
    dt = ImageData(image, backbone=backbone)
    return dt.to_tensor()


def img2tensor(file_path, backbone):
    """
    对图像文件进行数据预处理，返回推理机需要的数据类型
    :param file_path: 图像文件的地址
    :param backbone: 推理机的模型选择
    :return:
    """
    assert isinstance(file_path, str)
    assert isinstance(backbone, str)
    dt = ImageData(file_path, backbone=backbone)
    return dt.to_tensor


# 3. 推理函数

def infer_onnx(ckpt_path, input_data):
    """
    推理机的推理函数
    :param ckpt_path: onnx权重地址
    :param input_data: 仅数据预处理后的输入数据格式
    :return: 推理结果
    """
    sess = rt.InferenceSession(ckpt_path)
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    pred_onx = sess.run([out_name], {input_name: input_data})
    return pred_onx[0]
