# from playsound import playsound  
# from PIL import Image, ImageTk  
# import tkinter as tk  
# import random  
# import math  
# import numpy as np  
# import time  
# import sys  
# from collections import defaultdict  
# import itertools
# _fur_counter = itertools.count()

# # Fur 类表示猫毛
# class Fur:
#     def __init__(self, x, y):
#         self.x = x  # 猫毛的x坐标
#         self.y = y  # 猫毛的y坐标
#         self.size = 5
#         # >>> 新增：给每根毛一个唯一 tag，用于 later canvas.delete
#         self.tag = f"fur_{next(_fur_counter)}"
#         self.canvas_id = None
        
#     def draw(self, canvas):
#         """绘制猫毛"""
#         self.canvas_id = canvas.create_oval(
#             self.x - self.size, self.y - self.size,
#             self.x + self.size, self.y + self.size,
#             fill="yellow", tags=self.tag
#         )

#     def getLocation(self):
#         return self.x, self.y
    
#     def create_initial_fur(self, canvas):
#         #创建并绘制初始猫毛
#         # 创建一定数量的猫毛
#         #fur_count = random.randint(100, 200)  # 随机决定猫毛的数量
#         fur_count = 10
#         print(f"{self.name} is dropping initial fur...")  # Debug
#         for _ in range(fur_count):
#             # 随机生成猫毛的位置
#             fur_x = self.x + random.randint(-50, 50)
#             fur_y = self.y + random.randint(-50, 50)
#             fur = Fur(fur_x, fur_y)
#             fur.draw(canvas)  # 绘制猫毛 
    
#     def adjust_molt_probability(self):
#         """根据季节、情绪和速度调整掉毛概率"""
#         # 根据季节调整掉毛概率
#         if self.season in ["spring", "autumn"]:
#             self.molt_probability = 0.5
#         elif self.season == "winter":
#             self.molt_probability = 0.3
#         else:
#             self.molt_probability = 0.2

#         # 根据情绪和疲劳度调整掉毛概率
#         if self.personality == "erratic":
#             self.molt_probability += 0.2
#         if self.fatigue > 50:
#             self.molt_probability += 0.3

#         # 根据猫的运动速度调整掉毛概率
#         speed = (self.vl + self.vr) / 2
#         if speed > 5.0:
#             self.molt_probability += 0.3
#         elif speed > 2.0:
#             self.molt_probability += 0.1

#         # 限制掉毛概率最大为1.0
#         self.molt_probability = min(self.molt_probability, 1.0)

#     def draw(self, canvas):
#         """绘制猫毛"""
#         canvas.create_oval(self.x - self.size, self.y - self.size,
#                            self.x + self.size, self.y + self.size,
#                            fill="yellow", tags="fur")
#     def getLocation(self):
#         return self.x, self.y
    
#     def create_initial_fur(canvas, x, y, fur_count=10):
#         """在 (x,y) 周围随机撒 fur_count 根猫毛"""
#         for _ in range(fur_count):
#             offset_x = random.randint(-50, 50)
#             offset_y = random.randint(-50, 50)
#             fur = Fur(x + offset_x, y + offset_y)
#             fur.draw(canvas)

#         '''def drop_fur(self, canvas):
#         #根据掉毛概率掉毛
#         print(f"{self.name} is dropping fur!")
#         fur = Fur(self.x, self.y)
#         fur.draw(canvas)  # 绘制猫毛''' 
    
#     def drop_fur(self, canvas):
#         """根据掉毛概率随机在猫周围掉毛"""
#         if self.at_nest:  # 如果猫已经回到猫窝，就不再掉毛
#             return
#         self.molt_time_accumulated += 1

#         # 根据时间间隔调整掉毛概率
#         if self.molt_time_accumulated >= self.molt_time_interval:
#             if random.random() < self.molt_probability * self.molt_density_factor:
#                 offset_x = random.randint(-15, 15)
#                 offset_y = random.randint(-15, 15)
#                 fur = Fur(self.x + offset_x, self.y + offset_y)
#                 fur.draw(canvas)
#                 self.dropped_fur_count += 1
#                 if self.dropped_fur_count >= 100:
#                     self.return_to_nest(canvas)
#             self.molt_time_accumulated = 0.0
#             self.molt_density_factor = 1.0 + (math.sin(time.time() / 10.0) * 0.5)




# fur.py

import random
import math
import time
import itertools
from tkinter import Canvas  # 确保 Canvas 在作用域中

_fur_counter = itertools.count()

class Fur:
    """
    表示一根猫毛。每根猫毛有唯一的 tag，用于在画布上绘制和删除。
    """
    def __init__(self, x, y, size=5):
        self.x = x
        self.y = y
        self.size = size
        self.tag = f"fur_{next(_fur_counter)}"
        self.canvas_id = None

    def draw(self, canvas: Canvas):
        """
        在画布上绘制这根猫毛，并打上唯一的 tag。
        """
        self.canvas_id = canvas.create_oval(
            self.x - self.size, self.y - self.size,
            self.x + self.size, self.y + self.size,
            fill="yellow", tags=self.tag
        )

    def getLocation(self):
        """
        返回猫毛的 (x, y) 坐标。
        """
        return self.x, self.y


def adjust_molt_probability(cat):
    """
    根据猫的季节、性格、疲劳度和速度调整掉毛概率，
    并保存到 cat.molt_probability 属性。
    """
    season = getattr(cat, 'season', 'summer')
    # 季节基准概率
    if season in ["spring", "autumn"]:
        prob = 0.5
    elif season == "winter":
        prob = 0.3
    else:
        prob = 0.2

    # 性格影响
    if getattr(cat, 'personality', None) == 'erratic':
        prob += 0.2

    # 疲劳度影响
    if getattr(cat, 'fatigue', 0) > 50:
        prob += 0.3

    # 速度影响
    vl = getattr(cat, 'vl', 0)
    vr = getattr(cat, 'vr', 0)
    speed = (vl + vr) / 2
    if speed > 5.0:
        prob += 0.3
    elif speed > 2.0:
        prob += 0.1

    cat.molt_probability = min(prob, 1.0)


def drop_fur(cat, canvas: Canvas, passiveObjects: list):
    """
    当满足掉毛条件时，根据 cat.molt_probability 和密度因子
    在猫周围随机掉毛。生成的 Fur 对象会添加到 passiveObjects，
    并在掉毛数量达到上限后调用 cat.return_to_nest(canvas)。
    """
    # 如果猫已回窝，则不再掉毛
    if getattr(cat, 'at_nest', False):
        return passiveObjects

    # 先更新掉毛概率
    adjust_molt_probability(cat)

    # 累积时间
    cat.molt_time_accumulated = getattr(cat, 'molt_time_accumulated', 0) + 1
    interval = getattr(cat, 'molt_time_interval', 5)

    if cat.molt_time_accumulated >= interval:
        density = getattr(cat, 'molt_density_factor', 1.0)
        # 按概率决定是否掉毛
        if random.random() < cat.molt_probability * density:
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            fur = Fur(cat.x + dx, cat.y + dy)
            fur.draw(canvas)
            passiveObjects.append(fur)

            # 更新掉毛计数，超过限额就回窝
            cat.dropped_fur_count = getattr(cat, 'dropped_fur_count', 0) + 1
            if cat.dropped_fur_count >= getattr(cat, 'molt_limit', 100):
                cat.return_to_nest(canvas)

        # 重置计时器并更新密度因子
        cat.molt_time_accumulated = 0
        cat.molt_density_factor = 1.0 + math.sin(time.time() / 10.0) * 0.5
        
    return passiveObjects