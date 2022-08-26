from django.views.decorators.csrf import csrf_exempt
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.decorators import api_view
from cifar10_api.ml_models.conv_net.conv_net import ConvNetCifar10
from cifar10_api.ml_models.prediction import PredictionSerializer
from PIL import Image
import torch

# Create your views here.
@csrf_exempt
@api_view(['POST'])
def index(request: Request):
    file = request.FILES['img_file']
    img = Image.open(file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conv_net = ConvNetCifar10(device).to_device()
    pred = conv_net.predict(img)
    result = PredictionSerializer(pred)
    return Response({'prediction': result.data})