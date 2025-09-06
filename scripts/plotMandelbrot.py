import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./bin/release')  # 添加上级目录到系统路径

from mandelbrot import Mandelbrot


width = 1920
height = 1080
maxIterations = 8000

md = Mandelbrot(width, height, maxIterations)
md.pixelCalculation()
md.pixelCalculationOMP()
md.pixelCalculationCUDA()
data = md.getData()

data = np.array(data).reshape((height, width))

plt.figure(figsize=(width/100, height/100), dpi=100)
# 使用热力图绘制Mandelbrot集合
plt.imshow(data, cmap='hot_r', interpolation='bilinear')
# 隐藏坐标轴
plt.axis('off')
# 调整布局，确保图像占满整个画布
plt.tight_layout(pad=0)

plt.show()

# plt.savefig('mandelbrot.png', dpi=300, bbox_inches='tight', pad_inches=0)