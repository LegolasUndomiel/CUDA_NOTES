import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./bin/release')  # 添加上级目录到系统路径

from mandelbrot import Mandelbrot


width = 1920
height = 1080
maxIterations = 8000

md = Mandelbrot(width, height, maxIterations)
md.pixelCalculationCUDA()
data = md.getData()

data = np.array(data).reshape((height, width))

# 绘图功能补充
plt.figure(figsize=(width/100, height/100), dpi=100)
# 使用热力图绘制Mandelbrot集合
# 使用'hot'颜色映射并反转，使边缘更加突出
plt.imshow(data, cmap='hot_r', interpolation='bilinear')
# 隐藏坐标轴
plt.axis('off')
# 调整布局，确保图像占满整个画布
plt.tight_layout(pad=0)

# 显示图像
plt.show()

# 可选：保存图像到文件
# plt.savefig('mandelbrot.png', dpi=300, bbox_inches='tight', pad_inches=0)