import cv2
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D

#使用するPCのカメラを宣言(0:内カメラ, 1:外カメラ)
video_capture = cv2.VideoCapture(0)

#CNNモデルの作成
model = VGG16(weights='imagenet') #学習済みモデルVGG16の重みを使用したCNNモデルを構築
layers = model.layers[1:4] #"入力層-畳み込み層2つ-プーリング層1つ"だけ抽出
layer_outputs = [layer.output for layer in layers] #各層の出力
h_vgg16, w_vgg16 = 224, 224 # VGG16の入力画像サイズ
featureMap = 2 #特徴量マップの指定(0~63)


#モデルの構造を表示(正直いらない)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activation_model.summary()

while True:
    #フレーム情報を取得
    ret, frame = video_capture.read()
    
    #VGG16の入力が224×224なのでリサイズ(cv2.resizeだと縦横縮尺比が変)
    h_frame, w_frame = len(frame[:,0]), len(frame[0,:])
    center_h, center_w = int((h_frame / 2)), int((w_frame / 2))
    oneside_h, oneside_w = int((h_vgg16 / 2)), int((w_vgg16 / 2)) #oneside_h=112, oneside_w=112
    frame_ = frame[center_h - oneside_h : center_h + oneside_h, \
                   center_w - oneside_w : center_w + oneside_w]
    
    #入力画像に対する中間層の出力
    activations = activation_model.predict(frame_.reshape(1,h_vgg16,w_vgg16,3)) #activations.shape=(1,224,224,3)
    
    #畳み込み層の出力のみを抽出
    activations = [activation for layer, activation in zip(layers, activations) if isinstance(layer, Conv2D)] 
    
    #最初の畳み込み層の出力のみ抽出(activations[1][0]にすると2番目の畳み込み層の出力)
    activation = activations[0][0] #actuvation = (224,224,64)
    
    #指定した特徴量マップを出力(特徴量マップを変える時は featureMapの数値を変える:0~63)
    cv2.imshow('Video', activation[:,:,featureMap])
    
    #'q'を打ったら終了
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()# hasegawa_test
# hasegawa_test
