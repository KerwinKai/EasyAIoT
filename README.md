# EasyAIoT
## 简介
`EasyAIoT`是一款助力青少年物联网`消息通信`与`AI推理`的示例文档。通过集成`SIOT`、`EasyAPI`等工具实现上下位机之前的消息传递，通过`ONNXRuntime`推理引擎实现AI部署，在不改变原有学习的知识的基础上，增添新的参考函数，将传统的青少年物联网学科教育与AI推理结合起来。

## 示例代码
### 1. 消息通信
#### 1.1 SIoT通信方式
SIoT是一个针对学校场景的开源免费的MQTT服务器软件，可一键创建本地物联网服务器，摆脱联网困扰。
与Mind+结合可以让小学生到高中生都可以轻松上手物联网。其中，行空板已实现了SIoT的内置。使用详情可参考：[SIoT简介](https://mindplus.dfrobot.com.cn/siot)
* SIoT连接初始化
```angular2html
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
```
* SIoT消息发送
```angular2html
def publish_msg(IOT_pubTopic, msg, step_size=None):
    """
    功能：向siot上位机发送报文
    :param IOT_pubTopic: 设置发送主题
    :param msg: 需要发送的消息
    :param step_size: 设置是否需要分片传输，默认为整报文传送。面向资源极度受限的板子（售价约几十块，例如掌控板），
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

```
* SIoT消息监听
```angular2html
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
```
#### 1.2 EasyAPI通信方式
`EasyAPI`是`XEdu`团队基于`FastAPI`开发的接口启动与调用示例代码，摆脱联网与付费API的困扰。在`XEdu`一键安装包中内置了UI版本，这里提供部分代码，以实现文件通过EasyAPI传输的效果。
* EasyAPI的启动
```angular2html
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

```
* EasyAPI的调用：上传本地图片进行识别并返回结果

```angular2html
import requests
url = "http://localhost/upload" #需替换为生成的url地址
files = {'files': open('qingtian3.jpg', 'rb')}
result = requests.post(url=url, files=files)
print(result.text)
```

* EasyAPI的调用：CV2拍照后识别并返回结果
```angular2html
import cv2
import requests
cap = cv2.VideoCapture(0)
url = "http://localhost/upload" #需替换为生成的url地址
if(cap.isOpened()):
    ret_flag,Vshow = cap.read()#得到每帧图像
    save_url = 'save_data/1.jpg'
    cv2.imwrite(save_url,cv2.resize(Vshow,(224, 224)))
    files = {'files': open(save_url, 'rb')}
    result = requests.post(url=url, files=files)
    print(result.text)
else:
    print('摄像头未启动')
```

### 2. 文件预处理
要想将接收到的文件送入AI推理函数进行推理，还需将接收到的文件转换为tensor。
* 音频文件预处理
```angular2html
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
```
* 图像文件预处理
```angular2html
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
```

### 3. 推理函数
在推理设备上调用`ONNXRuntime`推理引擎进行模型的推理。
```angular2html
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
```

## 使用示例
### 案例一：掌控板+行空板实现语音识别
掌控板作为下位机，获取语音，通过siot协议（mqtt教育版）或urequests库对行空板启动的EasyAPI发送至行空板，行空板作为上位机，执行监听siot与API接口，语音转图像，onnx后端推理，发送执行命令等功能，结合传统物联网教学设计，形成一个完整的物联网语音读取，传输，识别和反馈的设计流程。

* 第一步：模型训练

参照[行空板上温州话识别](https://www.openinnolab.org.cn/pjlab/project?id=63b7c66e5e089d71e61d19a0&backpath=/pjlab/projects/list#public)项目，进行方言的数据集录制、训练与模型转换，得到ONNX模型。

* 第二步：上位机制作

将行空板作为上位机，需要启动SIoT与EasyAPI的接听，按逻辑设计UI。
```angular2html
import siot
import time
import binascii
import numpy as np
import BaseData
import librosa
import onnxruntime as rt
import matplotlib.pyplot as plt
from PIL import Image
import wave
import scipy.io.wavfile as wav
from matplotlib.cm import get_cmap
import scipy.signal as signal
from unihiker import GUI,Audio
import socket
from fastapi import FastAPI, UploadFile, File
import scipy.signal as signal
import uvicorn
import shutil

gui=GUI()  #实例化GUI类
audio = Audio() # 实例化Audio类

#gui 添加文本标题
gui.draw_text(x=60, y=20,color="#4169E1", font_size=18,text="方言识别助手")
gui.draw_text(x=60, y=60,color="#000000", font_size=10,text="通信模式：SIOT+EasyAPI")
gui.draw_text(x=20, y=100,color="#000000", font_size=10,text="状态：")
info_text = gui.draw_text(x=60, y=100, color="red",font_size=10,text="")
info_text_1 = gui.draw_text(x=60, y=140, color="red",font_size=10,text="")


sess = rt.InferenceSession("upload/mobilenetv2.onnx")

SERVER = "192.168.31.29"        #MQTT服务器IP
CLIENT_ID = ""            #在SIoT上，CLIENT_ID可以留空
IOT_recTopic  = 'ailab/sensor1'   #“topic”为“项目名称/设备名称”
IOT_pubTopic = 'ailab/sensor2'
IOT_UserName ='siot'        #用户名
IOT_PassWord ='dfrobot'     #密码
step_size = 4096
message_tol = b''
viridis_cmap = plt.get_cmap('viridis')                   
color_map = viridis_cmap.colors
color_map = (np.array(color_map) * 255).astype(np.uint8)
label = ["吃饭", "回家", "学校", "看电视", "睡觉"]

def onnx_infer(wave):
    fs = 16000
    if type(wave) == str :
        time1 = time.time()
        fs0, wave = wav.read(wave) # 读取原始采样频率和数据
        time2 = time.time()
        if fs0 != fs:        #如果采样频率不是16000则重采样
            time1 = time.time()
            num = int(len(wave) * fs/ fs0) # 计算目标采样点数
            wave = signal.resample(wave, num) # 对数据进行重采样
            time2 = time.time()
    time1 = time.time()
    spec = librosa.feature.melspectrogram(wave, sr=fs, n_fft=512)
    time2 = time.time()
    spec = librosa.power_to_db(spec, ref=np.max)
    
    spec_new = (((spec+80)/80)*255).astype(np.uint8)
    h,w = spec_new.shape
    rgb_matrix = np.array([color_map[i] for i in spec_new.flatten()]).reshape(h, w, 3)
    
    image = Image.fromarray(rgb_matrix.astype(np.uint8))
    image = np.array(image)
    dt = BaseData.ImageData(image,backbone = 'MobileNet')
    input_data = dt.to_tensor()
    
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    pred_onx = sess.run([out_name], {input_name:input_data})
    return pred_onx[0]

def pre_data():
    global message_tol
    print(len(message_tol))
    file_name = "output.wav"
    sample_width = 2  # 2字节的采样宽度
    sample_rate = 8000  # 采样率为44100Hz
    channels = 1  # 双声道
    with wave.open(file_name, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(message_tol)
    wav_file.close()
    
    ort_output = onnx_infer(file_name)
    idx = np.argmax(ort_output, axis=1)[0]
    print(idx, ort_output[0][idx])
    info_text.config(x=60, y=100, text='结果：'+ label[idx])
    info_text_1.config(x=60, y=140, text='置信度：' + str(round(ort_output[0][idx], 2)))
    str_pub = '结果：'+ label[idx] + ' 置信度：' + str(round(ort_output[0][idx], 2))
    siot.publish(IOT_pubTopic, str_pub)
    message_tol = b''

app = FastAPI()
@app.post("/upload")
async def upload_file(files: UploadFile = File(...)):
    info_text.config(x=60, y=100, text='接口被访问，正在识别...')
    info_text_1.config(x=60, y=140, text='')
    fileUpload = f"./{files.filename}"
    with open(fileUpload, "wb") as buffer:
        shutil.copyfileobj(files.file, buffer)
    ort_output = onnx_infer(fileUpload)
    idx = np.argmax(ort_output, axis=1)[0]
    print(idx, ort_output[0][idx])
    str_pub = '结果：'+ label[idx] + ' 置信度：' + str(round(ort_output[0][idx], 2))
    info_text.config(x=60, y=100, text='结果：'+ label[idx])
    info_text_1.config(x=60, y=140, text='置信度：' + str(round(ort_output[0][idx], 2)))
     
    return str_pub

def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def EasyAIoT(client, userdata, msg):
    info_text.config(x=60, y=100, text='SIOT正在接收消息...')
    info_text_1.config(x=60, y=140, text='')
    global message_tol
    topic = msg.topic
    payload = msg.payload.decode()
    receive_msg = binascii.a2b_base64(payload)
    if isinstance(receive_msg,bytes) and len(receive_msg) > 0:
        message_tol += receive_msg
        if len(receive_msg)!= step_size:
            info_text.config(x=60, y=100, text='接收消息完毕，正在识别...')
            info_text_1.config(x=60, y=140, text='')
            pre_data()
    else:
        message_tol = b''


if __name__ == '__main__':
    info_text.config(x=60, y=100, text='正在启动EasyAPI与SIOT')
    siot.init(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
    siot.connect()
    siot.loop()
    siot.subscribe(IOT_recTopic, EasyAIoT)
    try:
        ip = str(get_host_ip())
        print("接口地址为http://" + ip + "/upload")
        ip_str = "http://" + ip + "/upload"
        info_text.config(x=60, y=100, text='EasyAPI接口：')
        info_text_1.config(x=10, y=140, text=ip_str + 'SIOT已监听')
        uvicorn.run(app=app, host="0.0.0.0", port=80, workers=1)
    except:
        print("80端口已被占用，正在启用8089端口")
        ip = ip + ":8089"
        ip_str = "http://" + ip + "/upload"
        print("接口地址为http://" + ip + "/upload")
        info_text.config(x=60, y=100, text='EasyAPI接口：')
        info_text_1.config(x=10, y=140, text=ip_str + 'SIOT已监听')
        uvicorn.run(app=app, host="0.0.0.0", port=8089, workers=1)
```

* 第三步：下位机制作
将掌控板作为下位机，掌控板因为内存有限，一次不能录制并发送过长的音频，故在SIoT传输是采用分片传输的思想进行。
```angular2html
from mpython import *
import network
from siot import iot
import time
import audio
import urequests
import binascii
my_wifi = wifi()
my_wifi.connectWiFi("aicamp", "aicamp123")

SERVER = "192.168.31.29"            #MQTT服务器IP
CLIENT_ID = "XEdu"                  #在SIoT上，CLIENT_ID可以留空
IOT_pubTopic  = 'ailab/sensor1'       #“topic”为“项目名称/设备名称”
IOT_recTopic  = 'ailab/sensor2'
IOT_UserName ='siot'            #用户名
IOT_PassWord ='dfrobot'         #密码

audio_file = None
step_size = 4096
mode = 0
mode_zn = ['SIOT模式', 'EasyAPI模式']

g_siot_message = None
def _siot_callback(_siot_topic, _siot_msg):
    global __msg
    _siot_topic = str(_siot_topic, "utf-8")
    __msg = str(_siot_msg, "utf-8")
    if _siot_topic == IOT_recTopic: return _siot_callback_1(__msg)
    
def _siot_callback_1(_siot_msg):
    global audio_file
    global mode
    global g_siot_message
    g_siot_message = _siot_msg
    print(_siot_msg)
    oled.fill(0)
    oled.DispChar(str('语音识别：' + mode_zn[mode]), 0, 0, 1)
    oled.DispChar(str(g_siot_message), 0, 16, 1)
    oled.DispChar(str('按下 A键 重新开始录音'), 0, 32, 1)
    oled.DispChar(str('按下 B键 选择通信方式'), 0, 48, 1)
    oled.show()

def text_show(message):
    global g_siot_message
    g_siot_message = message
    oled.fill(0)
    oled.DispChar(str('语音识别：' + mode_zn[mode]), 0, 0, 1)
    oled.DispChar(str(g_siot_message), 0, 16, 1)
    oled.DispChar(str('按下 A键 重新开始录音'), 0, 32, 1)
    oled.DispChar(str('按下 B键 选择通信方式'), 0, 48, 1)
    oled.show()

def on_button_a_pressed(_):
    global audio_file
    oled.fill(0)
    oled.DispChar(str('正在录音，时长 1 秒 ...'), 0, 0, 1)
    oled.show()
    rgb[0] = (int(255), int(0), int(0))
    rgb.write()
    time.sleep_ms(1)
    audio.recorder_init(i2c)
    audio.record(audio_file, 1)
    audio.recorder_deinit()
    oled.fill(0)
    oled.DispChar(str('正在识别语音文字 ...'), 0, 0, 1)
    oled.DispChar(str('长时间未响应请检查'), 0, 16, 1)
    oled.DispChar(str('上位机连接并按A键重试'), 0, 32, 1)
    oled.show()
    rgb[0] = (int(51), int(255), int(51))
    rgb.write()
    time.sleep_ms(1)
    if mode:
        print('通过EasyAPI模式发送')
        _rsp = urequests.post("http://192.168.31.29:8089/upload", files={"files":(audio_file, "audio/wav")})
        print(_rsp.text)
        text_show(_rsp.text[1:-1])
    else:
        print('通过SIOT模式发送')
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
            f.write('')
        f.close()
        print(len(audio_bytes))
        for index in range(0, len(audio_bytes), step_size):
            b = binascii.b2a_base64(audio_bytes[index:min(index+step_size,len(audio_bytes))])
            siot.publish(IOT_pubTopic, b)
            time.sleep_ms(100)
        siot.getsubscribe(IOT_recTopic)

def on_button_b_pressed(_):
    global mode
    mode = 1 - mode
    if g_siot_message is not None:
        oled.fill_rect(0, 0, 128, 16, 0)
        oled.DispChar(str('语音识别：' + mode_zn[mode]), 0, 0, 1)
        oled.DispChar(str((g_siot_message)), 0, 16, 1)
        oled.DispChar(str('按下 A键 重新开始录音'), 0, 32, 1)
        oled.DispChar(str('按下 B键 选择通信方式'), 0, 48, 1)
        oled.show()
    else:
        oled.fill(0)
        oled.DispChar(str('语音识别：' + mode_zn[mode]), 0, 0, 1)
        oled.DispChar(str('按下 A键 开始录音'), 0, 16, 1)
        oled.DispChar(str('按下 B键 选择通信方式'), 0, 32, 1)
        oled.show()
    
button_a.event_pressed = on_button_a_pressed
button_b.event_pressed = on_button_b_pressed
siot = iot(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
siot.connect()
siot.loop()

siot.set_callback(_siot_callback)

audio_file = 'my.wav'

oled.fill(0)
oled.DispChar(str('语音识别：' + mode_zn[mode]), 0, 0, 1)
oled.DispChar(str('按下 A键 开始录音'), 0, 16, 1)
oled.DispChar(str('按下 B键 选择通信方式'), 0, 32, 1)
oled.show()
```

* 第四步：部署演示

在掌控板上按A键进行语音的录制，按B键进行通信模式的选择，详见：[演示视频](https://aicarrier.feishu.cn/docx/UzCCd5Kdpo9FJkxHI8ScLLkSnff#VCQ4dcs2iossSsxoNEQcYtDrnme)