import argparse 
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

#argparse: 명령줄 인자 처리하는 데 사용. ex: if __name__ == "__main__: "섹션에서 (cfg, batch-size, device 등)를 정의하고 사용자 입력을 받아서 처리
#contextlib: 컨텍스트 관리자를 쉽게 만들어주는 유틸리티 라이브러리. (NameError발생할 수 있을 때 contextlib.suppress로 예외를 처리함)
#math: 수학 관련 함수(math.log)
#os: 운영체제와 상호작용
#platform: 시스템 정보 제공. (코드가 윈도우에서 실행되는 지 여부 확인)
#sys: 파이썬 인터프리터와 상호 작용하는 표준 라이브러리. (sys.path에 프로젝트의 최상위 디렉토리를 추가하여 모듈을 가져 옴)
#deepcopy: 객체를 완전히 복사함(모델 구성을 복제하는 데 사용)
#Path: 경로 조작을 위한 기능 제공하는 모듈. (현재 파일 위치화 프로젝트 디렉토리 구조 처리)
#torch: 딥러닝 프레임워크인 PyTorch의 주요 모듈. (딥로닝 모델을 구축하고 춘련시키는 데 필요한 중요한 라이브러리임)
#torch.nn: PyTorch의 신경망 관련 모듈. 딥러닝 모델의 각 레이어를 정의하고 구현하는 데 사용됨

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#FILE = Path(__file__).resolve(): 현재 실행되고 있는 파일의 절대 경로를 FILE변수에 저장
#ROOT = FILE.parents[1] : 프로젝트의 루트 디렉토리를 나타내기 위해 현재 파일의 부모 디렉토리의 부모를(즉, 두 단계 위의 디렉토리를) ROOT변수에 저장한다고 정의.
#parents[1]은 parents[0]의 부모임. (현재 파일이 있는 디렉토리에서 두 단계 위의 디렉토리를 반환.)
#if str(ROOT) not in sys.path: : sys.path에 ROOT경로가 있는지 확인. (sys.path는 파이썬이 모듈 찾을 때 쓰는 거)
#sys.path.append(str(ROOT)) : ROOT경로가 sys.path에 없으면 추가해랴. (ROOT디렉토리에 있는 파일들을 파이썬 파일을 어디서든 import할 수 있게 하려고)
#if platform.system() != "Windows": : 현재 시스템이 Windows인지 확인
#ROOT = Path(os.path.relpath(ROOT, Path.cwd())): 윈도우가 아닌 경우, ROOT를 현재 작업 디렉토리(cwd)로부터 상대 경로로 변환하여 저장. 


from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)

from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

#모델 관련 모듈
#이해해야 하는 내용이 많아서 따로 txt파일 만들어 둠

try: 
    import thop
except ImportError:
    thop = None

#thop라이브러리를 import하고, 만약에 제대로 설치되지 않아 오류가 나더라고 thop변수를 None으로 설정하여 코드가 문제없이 작동하도록 처리하려는 블록이다.
#thop: 모델의 FLOPs를 계산하기 위한 라이브러리.
#FLOPs: Floating Point Operations per Second. 초당 부동 소수점 연산 횟수



class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    
    #stride: 구축과정에서 계산되는 변수. 각 피쳐 맵의 스트라이드를 의미하며, 기본적으로 이미지의 차원 감소율을 나타낸다.
    #입력 이미지에서 각 검출 레이어의 출력까지의 공간적 축소 비율이라고 함.
    #레이어의 앵커 박스 계산 시 사용되어, 특성 맵 상의 위치를 원본 이미지 상의 실제 위치로 변환하는 데 필요하다.abs

    #dynamic: True일 때, 네트워크가 추론 시 매번 그리드를 재구성하도록 강제하는 변수. 
    #그리드는 원래는 한 번 생성되면 변경되지 않지만, 입력 이미지의 크기가 달라지거나 다른 요인으로 인해 그리드의 재계산이 필요할 때 이 변수를 활성화하면 됨

    #export: 이 변수가 활성화되면 클래스는 추론 결과를 반환할 때 추가적인 정보나 중간 데이터 없이 최종 결과만 반환하는 방식으로 작동. 
    #이 변수는 다른 형식으로 변환하거나 내보낼 때 활성화하면 좋다고 함


    #Detect 클래스의 생성자를 정의하는 함수
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__() #부모 클래스의 생성자를 호출하는 방법. 
        #즉, nn.Module이 여기서는 부모 클래스 인 것이고, 이 부모 클래스의 모든 초기 설정이 수행됨. 이후 사용자가 정의한 네트워크 레이어나 모듈에 파이토치기능을 제공.
        
        self.nc = nc  # 모델이 인식해야 할 클래스의 수
        self.no = nc + 5  #각 앵커박스가 출력해야 하는 값의 총 수를 계산하여 self.no에 저장.
                          #nc + 5는 클래스 수에 대한 확률값들, 바운딩박스의 중심좌표, 너비와 높이, 객체 존재 확률을 포함한다.

        self.nl = len(anchors)  # anchors리스트의 길이를 사용하여 모델에 있는 검출 레이어의 수를 결정. 
                                #anchors: 각 레이어에 대한 앵커 박스의 크기 정의
        self.na = len(anchors[0]) // 2 # 첫 번째 앵커 그룹의 앵커 박스 개수 계산. 각 앵커는 너비와 높이(2개)를 가지므로, 전체 길이를 2로 나눈다.
        self.grid = [torch.empty(0) for _ in range(self.nl)]  #각 검출 레이어에 대해 초기화된 빈 그리드 리스트 생성. 
                                                              #이 때 생성된 그리드는 각 레이어의 특성 맵에 해당하는 위치 정보를 저장하는 데 사용 
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # anchor베열을 파이토치 텐서로 변환, 이를 모듈의 버퍼로 등록. float()으로 부동소수점 설정, view()로 각 레이어의 앵커를 적절한 형태로 재배열함
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


#위 함수부터 하면 됨