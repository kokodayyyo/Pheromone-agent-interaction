from playsound import playsound  
from PIL import Image, ImageTk  
import tkinter as tk  
import random  
import math  
import numpy as np  
import time  
import sys  
from collections import defaultdict  
# 顶部
from fur import Fur, adjust_molt_probability, drop_fur

MIN_X = 10
MAX_X = 990
MIN_Y = 10
MAX_Y = 990


selected_season = random.choice(["spring", "summer", "autumn", "winter"])
print(f"Randomly selected season: {selected_season}")

import os
from PIL import Image, ImageTk

class CatNest:
    def __init__(self, image_path=None):
        """初始化猫窝类，固定猫窝位置为(100, 600)"""
        self.x = 100  # 固定猫窝的x坐标
        self.y = 600  # 固定猫窝的y坐标
        self.image_path = image_path or os.path.join(os.getcwd(), 'catnet.jpg')  # 确保使用绝对路径
        self.image = None
        self.load_image()  # 加载图片并调整大小

    def load_image(self):
        """加载猫窝图片并调整大小为50x50"""
        try:
            img = Image.open(self.image_path)  # 尝试打开图片文件
            img = img.resize((50, 50), Image.LANCZOS)  # 调整图片的大小
            self.image = ImageTk.PhotoImage(img)  # 转换为Tkinter的图像格式
        except Exception as e:
            print(f"Error loading image: {e}")  # 如果加载失败，输出错误信息

    def draw(self, canvas):
        """在画布上绘制猫窝"""
        if self.image:
            canvas.create_image(self.x, self.y, image=self.image, tags="nest")  # 使用图片绘制猫窝
        else:
            # 如果图片加载失败，则使用默认的圆形猫窝
            width = 50  # 设置猫窝的宽度
            height = 50  # 设置猫窝的高度
            canvas.create_oval(self.x - width / 2, self.y - height / 2, self.x + width / 2, self.y + height / 2, fill="pink", tags="nest")


class PheromoneType:
    CHASE = 1  # 追逐型信息素
    ESCAPE = 2  # 逃避型信息素
    EXPLORE = 3  # 探索型信息素
    DECOY = 4  # 迷惑型信息素（新增）

# 空间哈希类，用于快速查找物体
class SpatialHash:
    def __init__(self, cell_size=50):
        # 初始化空间哈希的单元格大小
        self.cell_size = cell_size
        # 使用defaultdict创建一个空的字典来存储格子内的物体列表
        self.grid = defaultdict(list)

    # 添加物体到哈希网格中
    def add(self, x, y, obj):
        # 计算物体所在的格子坐标，格子的大小为self.cell_size
        cell_x, cell_y = int(x / self.cell_size), int(y / self.cell_size)
        # 将物体加入到对应格子的列表中
        self.grid[(cell_x, cell_y)].append(obj)

    def jump(self, big=False):
        """跳跃行为"""
        # 根据是否是大跳跃，随机生成跳跃的力量
        jump_power = random.randint(20, 50) if not big else random.randint(50, 100)
        # 随机生成跳跃的角度
        jump_angle = random.uniform(0, 2 * math.pi)

        # 根据跳跃力量和角度更新猫的位置
        self.x += jump_power * math.cos(jump_angle)
        self.y += jump_power * math.sin(jump_angle)

        # 处理环形空间，使猫在边界处循环
        if self.x < 0.0:
            self.x = 999.0
        if self.x > 1000.0:
            self.x = 0.0
        if self.y < 0.0:
            self.y = 999.0
        if self.y > 1000.0:
            self.y = 0.0

        # 删除画布上原来的猫
        self.canvas.delete(self.name)
        # 在新位置绘制猫
        self.draw(self.canvas)

        # 跳跃后短暂加速，设置左右速度为8.0
        self.vl = 8.0
        self.vr = 8.0

    # 获取指定位置周围一定半径范围内的物体
    def get_nearby(self, x, y, radius=1):
        # 创建一个空的集合来存储周围格子的坐标
        cells = set()
        # 计算覆盖范围内的格子数量（考虑到半径大小）
        radius_cells = int(radius / self.cell_size) + 1
        # 遍历周围的格子坐标
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                # 计算当前格子的坐标，并将其添加到cells集合中
                cells.add((int(x / self.cell_size) + dx, int(y / self.cell_size) + dy))
        
        # 返回在这些格子内找到的所有物体，遍历所有格子中的物体并返回
        return [obj for cell in cells for obj in self.grid.get(cell, [])]
    

# # Fur 类表示猫毛
# class Fur:
#     def __init__(self, x, y):
#         self.x = x  # 猫毛的x坐标
#         self.y = y  # 猫毛的y坐标
#         self.size = 5
        
#     # def draw(self, canvas):
    #     """绘制猫毛"""
    #     canvas.create_oval(self.x - self.size, self.y - self.size,
    #                        self.x + self.size, self.y + self.size,
    #                        fill="yellow", tags="fur")
    # def getLocation(self):
    #     return self.x, self.y
class Cat:
    def __init__(self, namep, canvasp,nestp):
        # 随机生成猫在画布上的初始x坐标，范围是100到900
        self.x = random.randint(100, 900)
        # 随机生成猫在画布上的初始y坐标，范围是100到900
        self.y = random.randint(100, 900)

        self.nest = nestp  # 确保这里是一个 CatNest 的实例

        # 随机生成猫的初始朝向角度，范围是0到2π
        self.theta = random.uniform(0.0, 2.0 * math.pi)
        # 给猫设置一个名字
        self.name = namep

        # 存储画布对象，用于后续在画布上绘制猫
        self.canvas = canvasp
        self.at_nest = False  # 初始化猫不在猫窝
        self.image = None

        # 新增: 控制掉毛的时间间隔和密集度
        self.molt_density_factor = 1.0  # 初始密集度因子
        self.molt_time_accumulated = 0.0  # 用于计算时间间隔
        self.molt_time_interval = random.uniform(2, 5)  # 掉毛的时间间隔，单位秒

        self.dropped_fur_count = 0  # 记录掉毛数量
        # 猫的左速度，初始值为1.0
        self.vl = 1.0
        # 猫的右速度，初始值为1.0
        self.vr = 1.0
        # 猫的转向状态，初始为0
        self.turning = 0
        # 猫的移动步数，随机范围在50到100之间
        self.moving = random.randrange(50, 100)
        # 猫当前是否正在转向，初始为False
        self.currentlyTurning = False
        # 猫的某种长度参数，初始值为20
        self.ll = 20
        # 记录上一次释放信息素的时间，初始为0
        self.last_pheromone_drop = 0
        # 猫的初始健康值为3
        self.health = 3
        # 随机选择猫的性格，有谨慎、大胆、多变三种
        self.personality = random.choice(['cautious', 'bold', 'erratic'])
        # 猫的疲劳度，初始为0
        self.fatigue = 0
        # 初始化空间哈希对象，用于空间管理
        self.spatial_hash = SpatialHash()

        self.season = selected_season  # 使用全局选择的季节
        adjust_molt_probability(self)

        # 现有的初始化代码...
        self.running = False  # 是否正在加速跑开
        self.run_distance = 400  # 加速跑开的目标距离
        self.run_speed = 8.0  # 加速跑开的速度
        self.run_angle = None  # 加速跑开的方向
        self.run_distance_remaining = 0  # 剩余需要跑的距离


        self.target_occupied = False

        # 尝试加载猫的图像
        try:
            # 打开名为cat.png的图像文件
            imgFile = Image.open("cat.png")
            # 将图像调整为30x30的大小，使用LANCZOS算法进行抗锯齿处理
            imgFile = imgFile.resize((30, 30), Image.LANCZOS)
            # 将PIL图像转换为Tkinter可用的图像对象
            self.image = ImageTk.PhotoImage(imgFile)
        except:
            # 如果图像加载失败，将图像对象设为None
            self.image = None

        # 添加猫毛
        #self.create_initial_fur(self.canvas)

    # def create_initial_fur(self, canvas):
    #     2#创建并绘制初始猫毛
    #     # 创建一定数量的猫毛
    #     #fur_count = random.randint(100, 200)  # 随机决定猫毛的数量
    #     fur_count = 10
    #     print(f"{self.name} is dropping initial fur...")  # Debug
    #     for _ in range(fur_count):
    #         # 随机生成猫毛的位置
    #         fur_x = self.x + random.randint(-50, 50)
    #         fur_y = self.y + random.randint(-50, 50)
    #         fur = Fur(fur_x, fur_y)
    #         fur.draw(canvas)  # 绘制猫毛 

   

    def draw(self, canvas):
        # 如果图像加载成功
        if self.image:
            # 在画布上绘制猫的图像，位置为(x, y)，并设置标签为猫的名字
            body = canvas.create_image(self.x, self.y, image=self.image, tags=self.name)
        else:
            # 如果图像加载失败，使用默认的图形绘制猫
            # 绘制一个灰色的圆形代表猫的身体
            body = canvas.create_oval(self.x - 15, self.y - 15, self.x + 15, self.y + 15,
                                      fill="gray", tags=self.name)
            # 绘制一条黑色的线来指示猫的朝向
            canvas.create_line(self.x, self.y,
                               self.x + 20 * math.cos(self.theta),
                               self.y + 20 * math.sin(self.theta),
                               fill="black", width=2, tags=self.name)

    # def adjust_molt_probability(self):
    #     """根据季节、情绪和速度调整掉毛概率"""
    #     # 根据季节调整掉毛概率
    #     if self.season in ["spring", "autumn"]:
    #         self.molt_probability = 0.5
    #     elif self.season == "winter":
    #         self.molt_probability = 0.3
    #     else:
    #         self.molt_probability = 0.2

    #     # 根据情绪和疲劳度调整掉毛概率
    #     if self.personality == "erratic":
    #         self.molt_probability += 0.2
    #     if self.fatigue > 50:
    #         self.molt_probability += 0.3

    #     # 根据猫的运动速度调整掉毛概率
    #     speed = (self.vl + self.vr) / 2
    #     if speed > 5.0:
    #         self.molt_probability += 0.3
    #     elif speed > 2.0:
    #         self.molt_probability += 0.1

    #     # 限制掉毛概率最大为1.0
    #     self.molt_probability = min(self.molt_probability, 1.0)



        
    def getLocation(self):
        # 返回猫当前的x和y坐标
        return self.x, self.y

    def thinkAndAct(self, agents, passiveObjects, canvas):
        print(f"{self.name} - Checking threat distance.")  # 添加输出确认方法运行
        # 更新空间哈希表，将所有智能体的位置信息添加进去
        self.spatial_hash = SpatialHash()
        for agent in agents:
            self.spatial_hash.add(agent.x, agent.y, agent)

        # 获取猫周围300单位距离内的物体
        nearby_objects = self.spatial_hash.get_nearby(self.x, self.y, 300)
        bots_nearby = []
        min_dist = float('inf')
        closest_bot = None

        # 遍历周围的物体，找出是Bot的物体，并记录它们的距离
        for obj in nearby_objects:
            if hasattr(obj, 'brain'):  # 是Bot
                distance = math.sqrt((obj.x - self.x) ** 2 + (obj.y - self.y) ** 2)
                if distance < 300:
                    bots_nearby.append((obj, distance))
                    if distance < min_dist:
                        min_dist = distance
                        closest_bot = obj
        print(f"{self.name} - Min distance to bot: {min_dist}.")  # 检查最小距离
        
        # 如果猫没有危险，则有掉毛行为
        if bots_nearby and min_dist > 200:  # 没有威胁，进行掉毛
            print(f"{self.name} - No threat detected. Proceeding to drop fur.")
            self.molt_time_accumulated += 0.1  # 假设每次调用thinkAndAct时时间间隔是0.1秒
            if self.molt_time_accumulated >= 5.0:
                passiveObjects = drop_fur(self, canvas, passiveObjects)
                self.molt_time_accumulated = 0.0  # 重置时间累计器
            
        else:
            print(f"{self.name} - Threat detected. Not dropping fur.")  # 有威胁，不掉毛
        
        # 根据猫的性格设置不同的反应距离和逃跑速度
        if self.personality == 'cautious':
            reaction_distance = 200
            escape_speed = 6.0
        elif self.personality == 'bold':
            reaction_distance = 150
            escape_speed = 8.0
        else:  # erratic
            reaction_distance = random.randint(100, 250)
            escape_speed = random.uniform(5.0, 9.0)

        # 如果有Bot靠近且距离小于反应距离，则执行逃跑行为
        if bots_nearby and min_dist < reaction_distance:
            # 增加疲劳度，最高到100
            self.fatigue = min(100, self.fatigue + 2)

            # 动态信息素释放策略
            current_time = time.time()
            # 根据距离决定信息素释放的频率，距离越近释放越频繁
            if current_time - self.last_pheromone_drop > max(0.5, 1.0 - min_dist / 400):
                # 计算真实逃跑信息素的强度，距离越近，强度越高
                pheromone_strength = 8 - (min_dist / 200) * 5
                # 向最近的Bot的大脑中添加逃跑信息素
                closest_bot.brain.add_pheromone(self.x, self.y, p_type=PheromoneType.ESCAPE,
                                                strength=pheromone_strength)

                # 根据性格决定是否释放迷惑性信息素
                if self.personality == 'erratic' or (self.personality == 'cautious' and random.random() < 0.3):
                    for _ in range(random.randint(1, 3)):
                        dx = random.uniform(-50, 50)
                        dy = random.uniform(-50, 50)
                        # 向最近的Bot的大脑中添加迷惑性信息素
                        closest_bot.brain.add_pheromone(self.x + dx, self.y + dy,
                                                        p_type=PheromoneType.DECOY,
                                                        strength=random.uniform(3, 6))

                self.last_pheromone_drop = current_time

            # 计算逃跑方向
            escape_direction = self.calculate_escape_direction(closest_bot, bots_nearby)

            # 考虑疲劳度对速度的影响，疲劳度越高速度越慢
            speed_modifier = max(0.5, 1.0 - self.fatigue / 200)

            # 转向逃跑方向
            angle_diff = (escape_direction - self.theta) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            if angle_diff > 0.3:
                self.vl = escape_speed * speed_modifier
                self.vr = -escape_speed * speed_modifier
            elif angle_diff < -0.3:
                self.vl = -escape_speed * speed_modifier
                self.vr = escape_speed * speed_modifier
            else:
                self.vl = escape_speed * speed_modifier
                self.vr = escape_speed * speed_modifier

            # 有5%的概率做出突然变向
            if random.random() < 0.05:
                self.vl *= -1
                self.vr *= -1
        else:
            # 没有危险时的漫游行为
            # 减少疲劳度，最低到0
            self.fatigue = max(0, self.fatigue - 1)

            if self.currentlyTurning:
                # 疲劳时转向更快
                turn_speed = 2.0 * (1.0 + self.fatigue / 100)
                self.vl = -turn_speed
                self.vr = turn_speed
                self.turning -= 1
            else:
                # 疲劳时移动更慢
                move_speed = 1.0 * (1.0 - self.fatigue / 200)
                self.vl = move_speed
                self.vr = move_speed
                self.moving -= 1

            # 如果移动步数为0且当前不在转向，则开始转向
            if self.moving == 0 and not self.currentlyTurning:
                self.turning = random.randrange(20, 40)
                self.currentlyTurning = True
            # 如果转向步数为0且当前在转向，则开始移动
            if self.turning == 0 and self.currentlyTurning:
                self.moving = random.randrange(50, 100)
                self.currentlyTurning = False

            # 有0.5%的概率随机改变行为模式
            if random.random() < 0.005:
                self.personality = random.choice(['cautious', 'bold', 'erratic'])
        return passiveObjects

    def calculate_escape_direction(self, closest_bot, bots_nearby):
        """计算最佳逃跑方向"""
        # 基本逃跑方向 - 远离最近的Bot
        dx = closest_bot.x - self.x
        dy = closest_bot.y - self.y
        escape_angle = math.atan2(dy, dx) + math.pi  # 相反方向

        # 如果有多个Bot在附近，计算平均逃跑方向
        if len(bots_nearby) > 1:
            avg_x = sum(bot.x for bot, _ in bots_nearby) / len(bots_nearby)
            avg_y = sum(bot.y for bot, _ in bots_nearby) / len(bots_nearby)
            avg_escape = math.atan2(avg_y - self.y, avg_x - self.x) + math.pi
            escape_angle = (escape_angle + avg_escape) / 2  # 取平均值

        # 考虑避开障碍物
        avoid_angle = self.avoid_obstacles()
        if avoid_angle is not None:
            escape_angle = (escape_angle + avoid_angle) / 2

        return escape_angle

    def avoid_obstacles(self):
        """避开障碍物"""
        # 定义16个方向
        directions = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        best_direction = None
        best_score = -float('inf')

        # 遍历每个方向，计算其得分
        for angle in directions:
            score = 0
            # 检查这个方向不同距离处是否畅通
            for dist in range(30, 151, 30):
                check_x = self.x + dist * math.cos(angle)
                check_y = self.y + dist * math.sin(angle)

                # 检查是否靠近边界
                if check_x < 50 or check_x > 950 or check_y < 50 or check_y > 950:
                    score -= 10

                # 检查是否有障碍物
                for obj in self.spatial_hash.get_nearby(check_x, check_y, 20):
                    if hasattr(obj, 'brain'):  # 是Bot
                        score -= 20 / max(1, dist)
                    else:  # 其他障碍物
                        score -= 10 / max(1, dist)

            # 如果方向接近当前方向，给予一定的偏好得分
            if abs(angle - self.theta) < math.pi / 4:
                score += 5

            # 更新最佳方向和得分
            if score > best_score:
                best_score = score
                best_direction = angle

        return best_direction

    def update(self, canvas, passiveObjects, dt):
        # 更新猫的位置，根据速度和时间间隔移动猫
        self.move(canvas, dt)

    def move(self, canvas, dt):
        if self.running:
            # 如果正在加速跑开
            if self.run_distance_remaining > 0:
                # 计算加速跑开的移动距离
                move_distance = self.run_speed * dt
                if move_distance > self.run_distance_remaining:
                    move_distance = self.run_distance_remaining

                # 更新位置
                self.x += move_distance * math.cos(self.run_angle)
                self.y += move_distance * math.sin(self.run_angle)

                # 减少剩余距离
                self.run_distance_remaining -= move_distance
            else:
                # 加速跑开完成
                self.running = False
                self.vl = 1.0  # 恢复正常速度
                self.vr = 1.0

        else:
            # 正常移动逻辑
            if self.vl == self.vr:
                newX = self.x + self.vr * math.cos(self.theta) * dt
                newY = self.y + self.vr * math.sin(self.theta) * dt
                newTheta = self.theta
            else:
                R = (self.ll / 2.0) * ((self.vr + self.vl) / (self.vl - self.vr))
                omega = (self.vl - self.vr) / self.ll
                ICCx = self.x - R * math.sin(self.theta)
                ICCy = self.y + R * math.cos(self.theta)

                m = np.matrix([[math.cos(omega * dt), -math.sin(omega * dt), 0],
                               [math.sin(omega * dt), math.cos(omega * dt), 0],
                               [0, 0, 1]])
                v1 = np.matrix([[self.x - ICCx], [self.y - ICCy], [self.theta]])
                v2 = np.matrix([[ICCx], [ICCy], [omega * dt]])
                newv = np.add(np.dot(m, v1), v2)

                newX = newv.item(0)
                newY = newv.item(1)
                newTheta = newv.item(2)

            newTheta = newTheta % (2.0 * math.pi)

            # 边界碰撞检测和反应
            collided = False
            if newX < MIN_X:
                newX = MIN_X
                newTheta = math.pi - newTheta
                collided = True
            elif newX > MAX_X:
                newX = MAX_X
                newTheta = math.pi - newTheta
                collided = True

            if newY < MIN_Y:
                newY = MIN_Y
                newTheta = -newTheta
                collided = True
            elif newY > MAX_Y:
                newY = MAX_Y
                newTheta = -newTheta
                collided = True

            if collided:
                self.vl *= 0.5
                self.vr *= 0.5
                newTheta = self.theta + random.uniform(-math.pi / 4, math.pi / 4)
                newTheta = newTheta % (2.0 * math.pi)
                print(f"{self.name} hit wall.")

            self.x = newX
            self.y = newY
            self.theta = newTheta

        # 删除画布上原来的猫
        canvas.delete(self.name)
        # 在新位置绘制猫
        self.draw(canvas)

    def jump(self, big=False):
        """跳跃行为"""
        jump_power = random.randint(60, 80) if not big else random.randint(90, 150)
        jump_angle = random.uniform(0, 2 * math.pi)

        potential_x = self.x + jump_power * math.cos(jump_angle)
        potential_y = self.y + jump_power * math.sin(jump_angle)
        final_x = max(MIN_X, min(potential_x, MAX_X))
        final_y = max(MIN_Y, min(potential_y, MAX_Y))
        self.x = final_x
        self.y = final_y

        # 删除画布上原来的猫
        self.canvas.delete(self.name)
        # 在新位置绘制猫
        self.draw(self.canvas)

        # 跳跃后短暂加速，设置左右速度为8.0
        self.vl = 8.0
        self.vr = 8.0

        # 启动加速跑开的行为
        self.running = True
        self.run_angle = jump_angle  # 沿着跳跃方向跑开
        self.run_distance_remaining = self.run_distance

    def return_to_nest(self, canvas):
        """让猫回到猫窝"""
        if self.nest:
            self.x = self.nest.x  # 设置猫的坐标为猫窝的坐标
            self.y = self.nest.y
            
            self.dropped_fur_count = 0  # 重置掉毛计数
            self.at_nest = True  # 标记猫已回到猫窝

            # 停止猫的运动（将速度设置为0）
            self.vl = 0
            self.vr = 0

            canvas.delete(self.name)  # 从画布上删除猫的图像
            self.draw(canvas)  # 在猫窝位置重新绘制猫

            # 禁止掉毛
            self.drop_fur = lambda x: None  # 禁用掉毛函数，防止猫回到猫窝后继续掉毛

    # '''def drop_fur(self, canvas):
    #     #根据掉毛概率掉毛
    #     print(f"{self.name} is dropping fur!")
    #     fur = Fur(self.x, self.y)
    #     fur.draw(canvas)  # 绘制猫毛''' 
    
    # def drop_fur(self, canvas):
    #     """根据掉毛概率随机在猫周围掉毛"""
    #     if self.at_nest:  # 如果猫已经回到猫窝，就不再掉毛
    #         return
    #     self.molt_time_accumulated += 1

    #     # 根据时间间隔调整掉毛概率
    #     if self.molt_time_accumulated >= self.molt_time_interval:
    #         if random.random() < self.molt_probability * self.molt_density_factor:
    #             offset_x = random.randint(-15, 15)
    #             offset_y = random.randint(-15, 15)
    #             fur = Fur(self.x + offset_x, self.y + offset_y)
    #             fur.draw(canvas)
    #             self.dropped_fur_count += 1
    #             if self.dropped_fur_count >= 100:
    #                 self.return_to_nest(canvas)
    #         self.molt_time_accumulated = 0.0
    #         self.molt_density_factor = 1.0 + (math.sin(time.time() / 10.0) * 0.5)

        

   