import matplotlib.pyplot as plt
#!nvcc --version
#!pip install mxnet-cu100
#!pip install gluoncv
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model

transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
net = get_model('cifar_resnet110_v1', classes=10, pretrained=True)


def detect(path):
    flag = False
    for imgs in os.listdir(path):
        if not imgs.endswith('.jpg'):
            continue
        im_path = os.path.join(path, imgs)

        img = image.imread(im_path)
        img = transform_fn(img)
        pred = net(img.expand_dims(axis=0))
        # print(pred)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        ind = nd.argmax(pred, axis=1).astype('int')
        if ind == 0 or ind == 2:
            flag = True
        '''
        print('The input picture is classified as [%s], with probability %.3f.'%
              (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))
        '''
    return flag


filepath = 'sample_data/images/'
flag = detect(filepath)
print(flag)