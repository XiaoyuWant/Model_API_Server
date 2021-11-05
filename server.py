import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
# System
import os
import time
from datetime import timedelta
from io import BytesIO
# Tools
import argparse
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # force read image file for trunctucated images
import timm # pip install timm, a opensource model structure library
from flask import make_response
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
def GetClassNames(mode):
    """
    get labels of all classes
    for difference situation
    """
    if mode == "debug":
        names=[]
        folders = glob.glob("/root/dataset/limit/*")
        for folder in folders:
            name = os.path.basename(folder)
            names.append(name)
    if mode == "hardcode":
        names=['苦瓜炒蛋', '芹菜肉丝', '南瓜粥', '木须肉', '蒜薹炒肉', '鹌鹑蛋烧肉', '红烧猪蹄', '宫保鸡丁', '糖醋里脊', '蒜香生菜', '地瓜糯米饼', '炒红苋菜', '硬皮糕点', '菠萝咕老豆腐', '炒莴苣叶', '芋头红烧肉', '紫菜蛋饺汤', '清淡丝瓜汤', '鳕鱼排', '焖鸡中翅', '奶油小西点', '西兰花番茄鸡蛋汤', '玉米胡萝卜排骨汤', '清炒黄瓜', '肉末豆腐', '鸡蛋饼', '银耳汤', '平菇肉片汤', '肉末豌豆', '清炒大白菜', '火腿蔬菜粥', '冬瓜排骨汤', '皮蛋冬瓜汤', '青椒炒肉', '酱黄瓜', '土豆芝士虾球', '蘑菇炒肉', '土豆牛腩', '煎薯条', '玉米胡萝卜汤', '炒空心菜', '肉末蒸蛋', '萝卜清汤', '卤鸡腿', '莴苣肉片', '家常肉末豆腐', '豆芽炒韭菜', '清炒丝瓜', '蚂蚁上树', '炒苕尖', '花卷', '白切鸡', '面包', '洋葱炒蛋', '蒜蓉空心菜', '肉丝烧茄子', '青菜肉片汤', '红烧丸子', '酥炸鱼排', '豆豉蒸排骨', '紫菜鸡蛋汤', '可可杏仁饼干', '可乐酱油鸡爪', '可乐鸡腿', '醋溜土豆丝', '擀面条', '腐竹烧肉', '萝卜泡菜', '煮饺子', '丝瓜蛋汤', '鲜蔬芙蓉汤', '瘦肉山药炒木耳', '虾仁豆腐汤', '蒜香油麦菜', '咖喱土豆牛肉焗饭', '清炒白菜', '豆角炖肉', '板栗烧肉', '红豆粥', '胡萝卜烧鸭', '鱼香茄子', '西红柿蛋汤', '雪菜土豆汤', '火腿炒蛋', '冬瓜肉片汤', '葱油拌刀削面', '鲜肉包', '台式卤肉饭', '清炒鸡毛菜', '肉丝炒白萝卜', '清炒娃娃菜', '丝瓜肉丸汤', '奶香小馒头', '土豆烧肉', '杏仁紫薯糕', '炒三丁', '香源 豆沙包', '炒南瓜', '鱼香肉丝', '咖喱鸡块', '热干面', '馄饨', '煎焗沙尖鱼', '韭菜炒鸡蛋', '糍粑鱼', '清炒花菜', '土豆炖牛腩', '胡萝卜木耳肉片', '虾米菠菜鸡蛋汤', '丝瓜炒蛋', '香干炒肉丝', '菠菜粉丝汤', '绿豆粥', '番茄鸡蛋疙瘩汤', '清炒江城菜心', '清炒豆芽', '豆浆', '玛格丽特饼干', '菠菜包', '番茄蛋花汤', 'Bibigo 海带汤', '土豆炖排骨', '排骨汤', '咖喱土豆炖鸡腿', '馒头', '黄瓜炒火腿肠', '肉末茄子', '水饺', '醋溜白菜', '葱油鲍鱼', '葱花花卷', '肉末粉丝', '番茄鸡蛋汤', '烧花蛤', '玉米炒火腿', '葱油花卷', '虎皮青椒', '煮麻辣牛肉', '番茄疙瘩汤', '杏鲍菇炒肉', '菠菜丸子汤', '醋溜大白菜', '红烧肉炖豆角', '手撕包菜', '玉米糊糊粥', '珍珠丸子', '皮蛋黄瓜汤', '水煮油麦菜', '胡萝卜黑木耳炒肉', '猪肉炖粉条', '烤茄子', '豆干素炒', '青椒肉丝', '萝卜丝饼', '青菜粥', '菠萝咕噜肉', '水煮肉片', '水煮鸡蛋', '莲子养生汤', '西红柿炒鸡蛋', '萝卜丝肉丸汤', '可乐排骨', '香菇炖鸡', '素炒花菜', '粉蒸排骨', '萝卜丝肉丸', '银耳绿豆汤', '虾仁炒西兰花', '红枣银耳汤', '冬瓜虾皮汤', '鸡米花', '韭黄炒蛋', '蒜蓉娃娃菜', '小米粥', '可可杂粮饼', '清炒土豆丝', '红烧豆腐', '菠萝炒饭', '莴笋胡萝卜肉片', '蒜蓉西兰花', '肉炒蒜苔', '菠萝古老肉', '豆腐青菜汤', '营养炒饭', '清炒小白菜', '粳米粥', '烤鲳鱼', '耗油生菜', '燕麦粥', '菌菇汤', '雪梨蒸冰糖', '玉米浓汤', '冬瓜虾米汤', '萝卜排骨汤', '虾仁青豆玉米', '清炒苋菜', '焖面', '萝卜烧牛腩', '红烧鸡翅', '肉末粉丝汤', '茄子豆角', '茶叶蛋', '粉蒸肉', '酱香热干面', '西蓝花炒鸡胸', '土豆焖肉', '可乐鸡翅', '荷包蛋焖面', '木耳枣鸡汤', '兴旺 雪菜', '千张炒肉丝', '清蒸南瓜', '糖炒栗子', '红烧鱼块', '清炒西兰花', '冬瓜炖牛肉', '牛奶煮麦片', '回锅肉', '西兰花肉片', '八记 红枣枸杞鸡汤', '香菇焖鸭', '烤虾', '红烧狮子头', '典发 千叶豆腐', '番茄龙利鱼', '土豆鲜肉饼', '黑米粥', '红烧排骨', '鸡蛋肉丝面条', '山药汤', '红豆汤', 'Fair Price 白馒头', '鸭血粉丝汤', '红薯粥', '绮罗园 苦荞粥', '榨菜', '凉拌海带丝', '猪骨红萝卜汤', '时蔬鸡蛋肉丝面', '银耳枸杞汤', '肉末毛豆', '酱豆腐（腐乳）', '芝麻糊', '果记 圣女果', '洋葱炒肉', '蚝油生菜', '豆腐肉丝汤', '滑蛋虾仁', '土豆烧鸡', '粉丝肉末汤', '青菜粉丝肉丸汤', '琥珀核桃', '山药羊肉汤', '红烧鸡腿', '西红柿炒菜花', '清炒南瓜', '清蒸龙利鱼柳', '牛肉萝卜汤', '豆奶', '葱油蚕豆', '香干回锅肉', '小馄饨', '葱爆牛肉', '清炒上海青', '清炒茼蒿菜', '南瓜汤', '烧腐竹', '苹果', '菌菇鱼片汤', '肉片烧菜花', '鹌鹑蛋红烧肉', '传统三鲜汤', '清炒黄豆芽', '紫菜蛋花汤', '烤香菇', '小笼包', '口味小龙虾', '葱香花卷', '西红柿炒蛋', '丝瓜肉丸子汤', '酸菜粉丝汤', '西芹肉丝', '清炒莴笋', '黑米馒头', '番茄炒蛋', '黄焖鸡', '青椒胡萝卜炒山药', '番茄豆皮汤', '芹菜炒肉', '豆花', '家常红烧糖醋排骨', '鱼丸汤', '绿豆汤', '梅林 梅干菜包', '梅干菜扣肉', '番茄豆芽汤', '佳诚 水果羹', '珍珠圆子', '干锅花菜', '砂锅杂粮粥']
    print(names)
    return names


def ImageTransform():
    """
    to get a image trainform of dataloader
    return: transform
    """
    transform = transforms.Compose([
        transforms.Resize(size=512),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    return transform

def InitModel():
    """
    initiate pretrained model
    and put into cuda device
    """
    model_path = "model_for_test.pt"
    NUM_CLASS = 300
    model = timm.create_model('tresnet_m_miil_in21k', pretrained=False,num_classes=NUM_CLASS)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    return model

def ImageToTensor(image,transform):
    """
    input: PIL.Image
    output: 4_dim tensor
    """
    tensor=transform(image) # 3_dim
    t = tensor.unsqueeze(0) # 4_dim
    return t

def GetImageTensorForInput(image,transform):
    """
    form a image file  to get a fitable input tensor
    """
    t = ImageToTensor(image,transform)
    return t

def GetInferenceResult(model,input,labels):
    """
    input: model tensor of image
    output: dict[label_name,probability] of top 3, SORTED form Big->Small
    """
    input = input.cuda()
    output = model(input)
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output)
    maxk = max((1,3))
    ret,predictions = output.topk(maxk,1,True,True)
    #   dim_1 of tensor to list
    ret = ret.squeeze(0).detach().cpu().numpy().tolist()
    predictions = predictions.squeeze(0).detach().cpu().numpy().tolist()

    result=[]
    for probability,class_num in zip(ret,predictions):
        result.append([labels[class_num],probability])
    result.sort(key=lambda x:x[1],reverse=True)
    return result

def SpeedTest(model,test_nums):
    """
    model is loaded in CUDA memory, test time should be very short
    ~ 0.026s/img 
    """
    image_path = "7.jpg"
    image=Image.open(image_path).convert("RGB")
    labels = GetClassNames(mode="hardcode") 
    tic_0 = time.time()
    for i in range(test_nums):
        tensor = GetImageTensorForInput(image=image,transform=transform)
        result = GetInferenceResult(model,tensor,labels)
    tic_1 = time.time()
    print("test for {} times, avarage time is {}".format(test_nums,(tic_1-tic_0)/test_nums))

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
app.config["JSON_AS_ASCII"]=False # ensure send utf-8 characters
@app.route("/upload",methods=['POST','GET'])
def upload():
    img = request.files.get('file').read()
    bytes_stream = BytesIO(img)
    img = Image.open(bytes_stream)
    tensor = GetImageTensorForInput(image=img,transform=transform)
    result = GetInferenceResult(model,tensor,labels)
    return result



if __name__=='__main__':
    # Load transform , model , label names 
    transform = ImageTransform()
    model = InitModel()
    labels = GetClassNames(mode="hardcode") 
    

    # This should be always in memory, and wait for request for inference .

    #TODO API for listen to network request?
    app.run(host="127.0.0.1",port=50001)

    # A image inference test
    # image_path = "7.jpg"
    # image=Image.open(image_path).convert("RGB")
    # tensor = GetImageTensorForInput(image=image,transform=transform)
    # result = GetInferenceResult(model,tensor,labels)

    # # Speed Test
    # SpeedTest(model,10)
   


