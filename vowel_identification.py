import numpy as np
import wave
import keras
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os,shutil



class my_model():
    def __init__(self,learning_rate,epochs,batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
    # 读取音频数据，转换成向量，长度为1s的采样次数。类内部调用
    def get_wav_mfcc(self,wav_path):
        f = wave.open(wav_path, 'rb')
        params = f.getparams()
        # print("params:",params)
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        waveData = np.reshape(waveData, [nframes, nchannels]).T
        f.close()
        # print('look here',waveData)
        ### 对音频数据进行长度大小的切割，保证每一个的长度都是一样的
        # 【因为训练文件全部是1秒钟长度，8000帧的，所以这里需要把每个语音文件的长度处理成一样的】
        data = list(np.array(waveData[0]))
        while len(data) > 8000:
            # del data[len(data)-1]
            del data[0]
        # print(len(data))
        while len(data) < 8000:
            data.append(0)
        # print(len(data))
        data = np.array(data)
        # 平方之后，开平方，取正数，值的范围在  0-1  之间
        data = data ** 2
        data = data ** 0.5
        return data

    # 加载数据集和标签[并返回标签集的处理结果],类内部调用
    def create_datasets(self):
        wavs=[]
        labels=[] # 训练集标签的名字   0：zero   1：one  2:two   3:three
        valiwavs=[]
        valilabels=[]# 验证集标签的名字   0：zero   1：one  2:two   3:three
        # 现在为了测试方便和快速直接写死，后面需要改成自动扫描文件夹和标签的形式
    #----------加载训练集------------
        path="train1\\zero\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            wavs.append(waveData)
            labels.append(0)
        path="train1\\one\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            wavs.append(waveData)
            labels.append(1)
        path="train1\\two\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            wavs.append(waveData)
            labels.append(2)
        path="train1\\three\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            wavs.append(waveData)
            labels.append(3)
    #----------加载验证集---------
        path="test\\zero\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            valiwavs.append(waveData)
            valilabels.append(0)
        path="test\\one\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            valiwavs.append(waveData)
            valilabels.append(1)
        path="test\\two\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            valiwavs.append(waveData)
            valilabels.append(2)
        path="test\\three\\"
        files = os.listdir(path)
        for file in files:
            waveData = self.get_wav_mfcc(path+file)
            valiwavs.append(waveData)
            valilabels.append(3)
        wavs=np.array(wavs)
        labels=np.array(labels)
        valiwavs=np.array(valiwavs)
        valilabels=np.array(valilabels)
        return (wavs,labels),(valiwavs,valilabels)
    #随机打乱数据顺序类内部调用
    def new_data(self,wavs,labels):
        new_wavs = np.zeros(shape=(wavs.shape[0],wavs.shape[1]))
        new_labels = np.zeros(shape=(labels.shape))
        k=0
        for i in np.random.permutation(len(labels)):
            new_wavs[k] = wavs[i,:]
            new_labels[k] = labels[i]
            k+=1
        return new_wavs,new_labels
    def build_model(self):#类内部调用
        model = models.Sequential()
        model.add(layers.Conv2D(8,(3,3),activation='relu',input_shape=(40,40,5)))
        model.add(layers.MaxPooling2D(2,2))
        model.add(layers.Conv2D(16,(3,3),activation='relu'))
        model.add(layers.MaxPooling2D(2,2))
        model.add(layers.Flatten())
        model.add(layers.Dense(4,activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=self.learning_rate),
                     metrics=['accuracy'])
        return model
    def train(self):#类外部调用
        model = self.build_model()
        (wavs,labels),(valiwavs,valilabels) = self.create_datasets()#加载数据
        new_wavs,new_labels = self.new_data(wavs,labels)#随机打乱数据顺序
        new_labels = to_categorical(new_labels)#one-hot
        valilabels = to_categorical(valilabels)
        #mean = np.mean(new_wavs)
        #std = np.std(new_wavs)
        #new_wavs=(new_wavs-mean)/std
        #testwavs=(testwavs-mean)/std
        train_wavs=new_wavs.reshape((4663,40,40,5))#转换成模型所需的输入张量大小
        train_labels=new_labels
        vali_wavs=valiwavs.reshape((1043,40,40,5))
        vali_labels=valilabels
        history = model.fit(train_wavs,train_labels,epochs=self.epochs,
                    batch_size=self.batch_size,validation_data=(vali_wavs,vali_labels))
        model.save('cnn_vowel_identification.h5')
        print('模型最终在训练集上的精度为：', history.history['acc'][-1])
        print('模型最终在验证集上的精度为：',history.history['val_acc'][-1])
        train_loss = history.history['loss']
        x=range(self.epochs)
        val_loss = history.history['val_loss']
        plt.plot(x, train_loss, 'b', label='train_loss')
        plt.plot(x, val_loss, 'r', label='val_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = my_model(0.001,40,32)
    model.train()
