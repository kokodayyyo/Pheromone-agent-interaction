from playsound import playsound
from PIL import Image, ImageTk
import tkinter as tk
import random
import math
import numpy as np
import time
import sys
import heapq
from collections import defaultdict


class PheromoneType:
    CHASE = 1
    ESCAPE = 2
    EXPLORE = 3
    DECOY = 4


class Pheromone:
    def __init__(self, p_type, strength, timestamp):
        self.type = p_type
        self.strength = strength
        self.timestamp = timestamp
        self.max_strength = strength  # 记录初始强度用于可视化


class Brain():
    def __init__(self, botp):
        self.bot = botp
        self.turningCount = 0
        self.movingCount = random.randrange(50, 100)
        self.currentlyTurning = False
        self.time = 0
        self.trainingSet = []
        self.dangerThreshold = 0

        # Improved pheromone system
        # 改进的信息素网格系统
        self.pheromone_grid = defaultdict(dict)  # 使用字典存储每个单元格的信息素
        self.cell_size = 20  # 网格大小
        self.max_pheromone_strength = 10  # 信息素最大强度
        self.pheromone_decay_rates = {
            PheromoneType.CHASE: 0.7,
            PheromoneType.ESCAPE: 0.8,
            PheromoneType.EXPLORE: 1,
            PheromoneType.DECOY: 0.8
        }
        self.last_pheromone_update = 0

        # Escape state management
        self.escape_state = "normal"
        self.escape_start_time = 0
        self.escape_target_angle = 0
        self.escape_direction = None

        # 不同类型信息素的衰减率和颜色
        self.pheromone_properties = {
            PheromoneType.CHASE: {'decay': 0.7, 'color': 'red'},
            PheromoneType.ESCAPE: {'decay': 0.8, 'color': 'blue'},
            PheromoneType.EXPLORE: {'decay': 1, 'color': 'green'},
            PheromoneType.DECOY: {'decay': 0.8, 'color': 'purple'}
        }

        # Dynamic programming table for cat tracking
        self.dp_table = {}  # Stores computed optimal actions
        self.dp_horizon = 5  # Planning horizon (steps ahead to consider)
        self.discount_factor = 0.9  # How much we value future rewards

        # Cat tracking parameters
        self.last_cat_positions = []  # Stores recent cat positions for prediction
        self.max_cat_history = 10  # How many past positions to remember

    def get_grid_cell(self, x, y):
        """Convert coordinates to grid cell coordinates"""
        return (int(x / self.cell_size), int(y / self.cell_size))

    def add_pheromone(self, x, y, p_type, strength=None):
        """添加信息素到网格中，支持叠加同类型信息素"""
        if strength is None:
            strength = {
                PheromoneType.CHASE: 5,
                PheromoneType.ESCAPE: 8,
                PheromoneType.EXPLORE: 3,
                PheromoneType.DECOY: 4
            }[p_type]

        # 限制最大强度
        strength = min(strength, self.max_pheromone_strength)

        cell_x, cell_y = self.get_grid_cell(x, y)
        timestamp = time.time()

        # 如果该位置已有同类型信息素，则叠加强度(不超过最大值)
        if p_type in self.pheromone_grid[(cell_x, cell_y)]:
            existing = self.pheromone_grid[(cell_x, cell_y)][p_type]
            new_strength = min(existing.strength + strength, self.max_pheromone_strength)
            self.pheromone_grid[(cell_x, cell_y)][p_type] = Pheromone(
                p_type, new_strength, timestamp)
        else:
            self.pheromone_grid[(cell_x, cell_y)][p_type] = Pheromone(
                p_type, strength, timestamp)

    def get_pheromones_in_cell(self, cell_x, cell_y, current_time, p_type=None):
        """获取特定单元格中的信息素，按类型过滤并应用衰减"""
        pheromones = []
        cell_pheromones = self.pheromone_grid.get((cell_x, cell_y), {})

        for p_type_key, pheromone in cell_pheromones.items():
            if p_type is None or p_type_key == p_type:
                decay_rate = self.pheromone_properties[p_type_key]['decay']
                elapsed = current_time - pheromone.timestamp
                decayed_strength = pheromone.strength * (decay_rate ** elapsed)
                if decayed_strength > 0.1:  # 忽略非常弱的信息素
                    pheromones.append((p_type_key, decayed_strength))

        return pheromones

    def update_pheromones(self):
        """更新信息素状态，应用衰减并清理过期信息素"""
        current_time = time.time()

        # 每0.5秒更新一次
        if current_time - self.last_pheromone_update < 0.5:
            return

        cells_to_delete = []

        for cell, pheromones in self.pheromone_grid.items():
            types_to_delete = []

            for p_type, pheromone in pheromones.items():
                # 应用类型特定的衰减率
                decay_rate = self.pheromone_properties[p_type]['decay']
                elapsed = current_time - pheromone.timestamp
                pheromone.strength *= (decay_rate ** elapsed)

                # 标记过弱的信息素待删除
                if pheromone.strength < 0.1:
                    types_to_delete.append(p_type)

            # 删除过弱的信息素
            for p_type in types_to_delete:
                del pheromones[p_type]

            # 如果单元格没有信息素了，标记待删除
            if not pheromones:
                cells_to_delete.append(cell)

        # 删除空单元格
        for cell in cells_to_delete:
            del self.pheromone_grid[cell]

        self.last_pheromone_update = current_time

    def draw_pheromones(self, canvas):
        """绘制信息素地图，不同类型不同颜色，强度决定透明度"""
        current_time = time.time()

        for (cell_x, cell_y), pheromones in self.pheromone_grid.items():
            if not pheromones:
                continue

            # 计算每个信息素类型的总强度
            type_strengths = defaultdict(float)
            for p_type, pheromone in pheromones.items():
                decay_rate = self.pheromone_properties[p_type]['decay']
                strength = pheromone.strength * (decay_rate ** (current_time - pheromone.timestamp))
                type_strengths[p_type] += strength

            # 按强度排序，先绘制弱的信息素
            sorted_types = sorted(type_strengths.items(), key=lambda x: x[1])

            for p_type, strength in sorted_types:
                if strength < 0.5:  # 忽略过弱的信息素
                    continue

                # 计算透明度
                alpha = min(0.7, strength / self.max_pheromone_strength)
                color = self.pheromone_properties[p_type]['color']

                # 创建带透明度的颜色
                if color.startswith('#'):
                    rgb = color[1:]
                    fill_color = f"#{rgb}%02x" % int(alpha * 255)
                else:
                    # 对于命名颜色，使用stipple模式模拟透明度
                    fill_color = color
                    stipple = "gray50" if alpha < 0.5 else "gray25"

                # 绘制信息素单元格
                x1 = cell_x * self.cell_size
                y1 = cell_y * self.cell_size
                x2 = (cell_x + 1) * self.cell_size
                y2 = (cell_y + 1) * self.cell_size

                if color.startswith('#'):
                    canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=fill_color,
                        outline="",
                        tags="pheromone"
                    )
                else:
                    canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=fill_color,
                        stipple=stipple,
                        tags="pheromone"
                    )

    def calculate_pheromone_gradient(self, x, y, current_time, p_type):
        """计算信息素梯度方向"""
        cell_x, cell_y = self.get_grid_cell(x, y)
        max_strength = 0
        best_angle = None

        # 检查周围3x3区域
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # 跳过中心单元格

                check_cell = (cell_x + dx, cell_y + dy)
                pheromones = self.get_pheromones_in_cell(check_cell[0], check_cell[1],
                                                         current_time, p_type)

                for _, strength in pheromones:
                    if strength > max_strength:
                        max_strength = strength
                        # 计算朝向信息素的方向
                        target_x = check_cell[0] * self.cell_size + self.cell_size / 2
                        target_y = check_cell[1] * self.cell_size + self.cell_size / 2
                        best_angle = math.atan2(target_y - y, target_x - x)

        return best_angle

    def calculate_escape_direction(self, x, y):
        """基于ESCAPE信息素计算逃跑方向"""
        current_time = time.time()
        cell_x, cell_y = self.get_grid_cell(x, y)

        best_angle = self.bot.theta  # 默认使用当前方向
        max_strength = 0

        # 检查当前单元格周围3x3区域
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell_x + dx, cell_y + dy)
                pheromones = self.get_pheromones_in_cell(check_cell[0], check_cell[1],
                                                         current_time, PheromoneType.ESCAPE)

                for p_type, strength in pheromones:
                    if strength > max_strength:
                        max_strength = strength
                        # 计算远离此信息素的方向
                        target_x = check_cell[0] * self.cell_size + self.cell_size / 2
                        target_y = check_cell[1] * self.cell_size + self.cell_size / 2
                        best_angle = math.atan2(target_y - y, target_x - x) + math.pi  # 相反方向

        return best_angle

    def update_escape_state(self, x, y, current_time):
        """处理逃跑状态机"""
        cell_x, cell_y = self.get_grid_cell(x, y)
        escape_pheromones = self.get_pheromones_in_cell(cell_x, cell_y, current_time, PheromoneType.ESCAPE)
        escape_detected = any(strength > 3.0 for _, strength in escape_pheromones)

        if self.escape_state == "normal" and escape_detected:
            self.escape_state = "pausing"
            self.escape_start_time = current_time
            self.escape_direction = None
            return True  # 需要立即停止

        elif self.escape_state == "pausing":
            if current_time - self.escape_start_time >= 2.0:
                self.escape_state = "tracking"
                self.escape_target_angle = self.calculate_escape_direction(x, y)
                return False
            return True  # 仍在暂停中

        elif self.escape_state == "tracking":
            if current_time - self.escape_start_time >= 7.0:
                self.escape_state = "normal"
            return False

        return False

    def predict_cat_movement(self, current_cat_position):
        """Predict future cat positions using linear extrapolation"""
        if len(self.last_cat_positions) < 2:
            return current_cat_position  # Not enough data for prediction

        # Calculate average velocity
        dx = sum(p[0] - self.last_cat_positions[i - 1][0]
                 for i, p in enumerate(self.last_cat_positions[1:], 1)) / (len(self.last_cat_positions) - 1)
        dy = sum(p[1] - self.last_cat_positions[i - 1][1]
                 for i, p in enumerate(self.last_cat_positions[1:], 1)) / (len(self.last_cat_positions) - 1)

        # Extrapolate position
        predicted_x = current_cat_position[0] + dx * self.dp_horizon
        predicted_y = current_cat_position[1] + dy * self.dp_horizon
        return (predicted_x, predicted_y)

    def dp_tracking_value(self, state, step):
        """Dynamic programming value function for cat tracking"""
        # Check if we've already computed this state
        state_key = (tuple(state['robot_pos']), state['robot_theta'],
                     tuple(state['cat_pos']), step)
        if state_key in self.dp_table:
            return self.dp_table[state_key]

        # Base case: reached planning horizon
        if step >= self.dp_horizon:
            return {'value': 0, 'action': (0, 0)}

        # Calculate immediate reward (closer to cat is better)
        distance = math.sqrt((state['robot_pos'][0] - state['cat_pos'][0]) ** 2 +
                             (state['robot_pos'][1] - state['cat_pos'][1]) ** 2)
        immediate_reward = -distance  # Negative because we want to minimize distance

        # Generate possible actions (simplified for efficiency)
        possible_actions = [
            (2.0, 2.0),  # Forward
            (3.0, -3.0),  # Right turn
            (-3.0, 3.0),  # Left turn
            (0.0, 0.0)  # Stop
        ]

        best_value = -float('inf')
        best_action = (0, 0)

        # Evaluate each possible action
        for action in possible_actions:
            # Simulate next state
            new_theta = state['robot_theta']
            new_pos = list(state['robot_pos'])

            if action[0] == action[1]:  # Moving straight
                new_pos[0] += action[0] * math.cos(new_theta)
                new_pos[1] += action[0] * math.sin(new_theta)
            else:  # Turning
                R = (self.bot.ll / 2.0) * ((action[1] + action[0]) / (action[0] - action[1]))
                omega = (action[0] - action[1]) / self.bot.ll
                ICCx = state['robot_pos'][0] - R * math.sin(new_theta)
                ICCy = state['robot_pos'][1] + R * math.cos(new_theta)

                # Simplified rotation (approximation)
                new_theta += omega
                new_theta %= (2 * math.pi)

            # Predict cat movement
            predicted_cat_pos = self.predict_cat_movement(state['cat_pos'])

            # Recursive call for future value
            next_state = {
                'robot_pos': new_pos,
                'robot_theta': new_theta,
                'cat_pos': predicted_cat_pos
            }
            future_value = self.dp_tracking_value(next_state, step + 1)['value']

            # Total value is immediate reward plus discounted future value
            total_value = immediate_reward + self.discount_factor * future_value

            if total_value > best_value:
                best_value = total_value
                best_action = action

        # Store computed value and action
        self.dp_table[state_key] = {'value': best_value, 'action': best_action}
        return self.dp_table[state_key]

    def optimal_tracking_action(self, robot_x, robot_y, robot_theta, cat_x, cat_y):
        """Compute optimal action using dynamic programming"""
        # Update cat position history
        if len(self.last_cat_positions) >= self.max_cat_history:
            self.last_cat_positions.pop(0)
        self.last_cat_positions.append((cat_x, cat_y))

        # Initialize DP state
        initial_state = {
            'robot_pos': (robot_x, robot_y),
            'robot_theta': robot_theta,
            'cat_pos': (cat_x, cat_y)
        }

        # Compute optimal action
        result = self.dp_tracking_value(initial_state, 0)
        return result['action']


    def thinkAndAct(self, lightL, lightR, chargerL, chargerR, x, y, sl, sr, battery, camera, collision):
        import time
        theta = self.bot.theta
        current_time = time.time()

        # Update pheromone states
        self.update_pheromones()

        # 定期释放EXPLORE信息素
        if self.time % 50 == 0:  # 每50个时间步释放一次
            self.add_pheromone(x, y, p_type=PheromoneType.EXPLORE, strength=3)

        # Training phase logic
        dangerDetected = False
        trainingTime = 1000
        if self.time < trainingTime:
            self.trainingSet.append((camera, collision))
        elif self.time == trainingTime:
            warningValues = []
            for i, tt in enumerate(self.trainingSet):
                if i >= 5 and tt[1] == True:
                    warningValues.append(self.trainingSet[i - 5][0])
            countWV = 0
            sumWV = 0
            for wv in warningValues:
                if not wv == [0] * 9:
                    sumWV += max(wv)
                    countWV += 1
            if countWV > 0:
                self.dangerThreshold = sumWV / countWV
            else:
                self.dangerThreshold = 0.5
        elif self.time > trainingTime:
            if any(c > self.dangerThreshold for c in camera):
                dangerDetected = True

        self.time += 1

        # Initialize state
        newX = None
        newY = None
        speedLeft = 0.0
        speedRight = 0.0

        # 检查ESCAPE信息素并计算梯度
        cell_x, cell_y = self.get_grid_cell(x, y)
        escape_pheromones = self.get_pheromones_in_cell(cell_x, cell_y, current_time, PheromoneType.ESCAPE)
        escape_detected = any(strength > 3.0 for _, strength in escape_pheromones)

        # 计算ESCAPE信息素梯度方向
        escape_gradient_angle = self.calculate_pheromone_gradient(x, y, current_time, PheromoneType.ESCAPE)

        # State transition logic
        if self.escape_state == "normal" and escape_detected:
            self.escape_state = "pausing"
            self.escape_start_time = current_time
            self.escape_direction = None
        elif self.escape_state == "pausing" and current_time - self.escape_start_time >= 2.0:
            self.escape_state = "tracking"
            self.escape_target_angle = self.calculate_escape_direction(x, y)
        elif self.escape_state == "tracking" and current_time - self.escape_start_time >= 7.0:
            self.escape_state = "normal"

        # Behavior priority handling (highest to lowest)

        # 1. Highest priority: Charging needs
        if battery < 600:
            if chargerR > chargerL:
                speedLeft, speedRight = 2.0, -2.0
            elif chargerR < chargerL:
                speedLeft, speedRight = -2.0, 2.0
            if abs(chargerR - chargerL) < chargerL * 0.1:
                speedLeft, speedRight = 5.0, 5.0
        if chargerL + chargerR > 200 and battery < 1000:
            speedLeft, speedRight = 0.0, 0.0

        # 2. High priority: Escape state handling
        if self.escape_state == "pausing":
            # Paused state - completely stop
            speedLeft, speedRight = 0.0, 0.0
        elif self.escape_state == "tracking":
            # Use dynamic programming for optimal tracking
            if any(c > 0.5 for c in camera):  # If cat is visible
                max_idx = camera.index(max(camera))
                # Estimate cat position based on camera reading
                cat_distance = 400 * (1 - max(camera))  # Rough distance estimate
                cat_angle = (max_idx - 4) * 0.2
                cat_x = x + cat_distance * math.cos(theta + cat_angle)
                cat_y = y + cat_distance * math.sin(theta + cat_angle)

                # Get optimal action from DP
                speedLeft, speedRight = self.optimal_tracking_action(x, y, theta, cat_x, cat_y)
            else:
                # Fall back to pheromone tracking if cat not visible
                angle_diff = (self.escape_target_angle - theta) % (2 * math.pi)
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi

                if abs(angle_diff) > 0.2:  # Need to turn
                    turn_speed = 3.0
                    if angle_diff > 0:
                        speedLeft, speedRight = turn_speed, -turn_speed
                    else:
                        speedLeft, speedRight = -turn_speed, turn_speed
                else:  # Correct direction, move forward
                    speedLeft, speedRight = 2.0, 2.0

        # 3. 如果没有被更高优先级行为覆盖，执行常规行为
        if speedLeft == 0.0 and speedRight == 0.0:
            # 检测并追逐猫
            cat_detected = any(c > 0.5 for c in camera)
            if cat_detected:
                max_idx = camera.index(max(camera))
                angle = (max_idx - 4) * 0.2

                if abs(angle) > 0.2:  # 需要转向
                    turn_speed = 2.0
                    if angle > 0:
                        speedLeft, speedRight = turn_speed, -turn_speed
                    else:
                        speedLeft, speedRight = -turn_speed, turn_speed
                else:  # 方向正确，前进
                    speedLeft, speedRight = 2.0, 2.0

                # 添加CHASE信息素
                self.add_pheromone(x, y, p_type=PheromoneType.CHASE, strength=5)

            # 如果没有检测到猫，但检测到ESCAPE信息素，则追踪
            elif escape_detected and escape_gradient_angle is not None:
                angle_diff = (escape_gradient_angle - theta) % (2 * math.pi)
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi

                if abs(angle_diff) > 0.5:  # 需要转向
                    turn_speed = 3.0
                    if angle_diff > 0:
                        speedLeft, speedRight = turn_speed, -turn_speed
                    else:
                        speedLeft, speedRight = -turn_speed, turn_speed
                else:  # 方向正确，前进
                    speedLeft, speedRight = 4.0, 4.0

                # 添加CHASE信息素
                self.add_pheromone(x, y, p_type=PheromoneType.CHASE, strength=3)
            else:
                # 信息素梯度追踪 - 现在会避让EXPLORE信息素
                # 在thinkAndAct方法中替换信息素引导部分
                max_grad = 0
                best_angle = 0
                min_explore = float('inf')
                best_explore_angle = 0

                for d_angle in np.linspace(-math.pi / 2, math.pi / 2, 9):
                    check_x = x + 50 * math.cos(theta + d_angle)
                    check_y = y + 50 * math.sin(theta + d_angle)
                    cell_x, cell_y = self.get_grid_cell(check_x, check_y)

                    # 获取所有信息素
                    cell_pheromones = self.pheromone_grid.get((cell_x, cell_y), {})
                    attractive = 0
                    repulsive = 0

                    for p_type, pheromone in cell_pheromones.items():
                        if p_type in (PheromoneType.CHASE, PheromoneType.ESCAPE):
                            attractive += pheromone.strength
                        elif p_type == PheromoneType.EXPLORE:
                            repulsive += pheromone.strength * 0.7  # 探索信息素有70%的排斥效果
                        elif p_type == PheromoneType.DECOY:
                            attractive += pheromone.strength * 0.3  # 迷惑信息素只有30%的吸引力

                    net_strength = attractive - repulsive

                    if net_strength > max_grad:
                        max_grad = net_strength
                        best_angle = d_angle

                    if repulsive < min_explore:
                        min_explore = repulsive
                        best_explore_angle = d_angle

                if max_grad > 1:  # 存在有效梯度
                    if abs(best_angle) > 0.3:
                        turn_speed = 4.0
                        if best_angle > 0:
                            speedLeft, speedRight = turn_speed, -turn_speed
                        else:
                            speedLeft, speedRight = -turn_speed, turn_speed
                    else:
                        speedLeft, speedRight = 5.0, 5.0
                elif min_explore > 2:  # 如果周围有探索信息素，避让它们
                    if abs(best_explore_angle) > 0.3:
                        turn_speed = 3.0
                        if best_explore_angle > 0:
                            speedLeft, speedRight = -turn_speed, turn_speed  # 反向转向
                        else:
                            speedLeft, speedRight = turn_speed, -turn_speed
                    else:
                        speedLeft, speedRight = 4.0, 4.0
                else:
                    # 默认漫游行为
                    if self.currentlyTurning:
                        speedLeft, speedRight = -2.0, 2.0
                        self.turningCount -= 1
                    else:
                        speedLeft, speedRight = 4.0, 4.0
                        self.movingCount -= 1

                    if self.movingCount == 0 and not self.currentlyTurning:
                        self.turningCount = random.randrange(20, 40)
                        self.currentlyTurning = True
                    if self.turningCount == 0 and self.currentlyTurning:
                        self.movingCount = random.randrange(50, 100)
                        self.currentlyTurning = False

        # Toroidal space handling
        if x > 1000:
            newX = 0
        if x < 0:
            newX = 1000
        if y > 1000:
            newY = 0
        if y < 0:
            newY = 1000

        return speedLeft, speedRight, newX, newY, dangerDetected

# 全局变量定义
MIN_X = 10
MAX_X = 990
MIN_Y = 10
MAX_Y = 990

class Bot():
    def __init__(self, namep, canvasp):
        self.name = namep
        self.canvas = canvasp
        self.x = random.randint(100, 900)
        self.y = random.randint(100, 900)
        self.theta = random.uniform(0.0, 2.0 * math.pi)
        self.ll = 60  # axle width
        self.sl = 0.0
        self.sr = 0.0
        self.battery = 1000



    def thinkAndAct(self, agents, passiveObjects, canvas):
        lightL, lightR = self.senseLight(passiveObjects)
        chargerL, chargerR = self.senseChargers(passiveObjects)
        collision = self.collision(agents)

        view = self.look(agents)
        self.sl, self.sr, xx, yy, dangerDetected = self.brain.thinkAndAct \
            (lightL, lightR, chargerL, chargerR, self.x, self.y, \
             self.sl, self.sr, self.battery, view, collision)
        if xx != None:
            self.x = xx
        if yy != None:
            self.y = yy

    def setBrain(self, brainp):
        self.brain = brainp

    def look(self, agents):
        from cat import Cat
        self.view = [0] * 9
        for idx, pos in enumerate(self.cameraPositions):
            for cc in agents:
                if isinstance(cc, Cat):
                    dd = self.distanceTo(cc)
                    scaledDistance = max(400 - dd, 0) / 400
                    ncx = cc.x - pos[0]  # cat if robot were at 0,0
                    ncy = cc.y - pos[1]
                    m = math.tan(self.theta)
                    A = m * m + 1
                    B = 2 * (-m * ncy - ncx)
                    r = 15  # radius
                    C = ncy * ncy - r * r + ncx * ncx
                    if B * B - 4 * A * C >= 0 and scaledDistance > self.view[idx]:
                        self.view[idx] = scaledDistance
        self.canvas.delete("view")
        for vv in range(9):
            if self.view[vv] == 0:
                self.canvas.create_rectangle(850 + vv * 15, 50, 850 + vv * 15 + 15, 65, fill="white", tags="view")
            if self.view[vv] > 0:
                colour = hex(15 - math.floor(self.view[vv] * 16.0))  # scale to 0-15 -> hex
                fillHex = "#" + colour[2] + colour[2] + colour[2]
                self.canvas.create_rectangle(850 + vv * 15, 50, 850 + vv * 15 + 15, 65, fill=fillHex, tags="view")
        return self.view

    def senseLight(self, passiveObjects):
        from others import Lamp
        lightL = 0.0
        lightR = 0.0
        for pp in passiveObjects:
            if isinstance(pp, Lamp):
                lx, ly = pp.getLocation()
                distanceL = math.sqrt((lx - self.sensorPositions[0]) * (lx - self.sensorPositions[0]) + \
                                      (ly - self.sensorPositions[1]) * (ly - self.sensorPositions[1]))
                distanceR = math.sqrt((lx - self.sensorPositions[2]) * (lx - self.sensorPositions[2]) + \
                                      (ly - self.sensorPositions[3]) * (ly - self.sensorPositions[3]))
                lightL += 200000 / (distanceL * distanceL)
                lightR += 200000 / (distanceR * distanceR)
        return lightL, lightR

    def senseChargers(self, passiveObjects):
        from others import Charger
        chargerL = 0.0
        chargerR = 0.0
        for pp in passiveObjects:
            if isinstance(pp, Charger):
                lx, ly = pp.getLocation()
                distanceL = math.sqrt((lx - self.sensorPositions[0]) * (lx - self.sensorPositions[0]) + \
                                      (ly - self.sensorPositions[1]) * (ly - self.sensorPositions[1]))
                distanceR = math.sqrt((lx - self.sensorPositions[2]) * (lx - self.sensorPositions[2]) + \
                                      (ly - self.sensorPositions[3]) * (ly - self.sensorPositions[3]))
                chargerL += 200000 / (distanceL * distanceL)
                chargerR += 200000 / (distanceR * distanceR)
        return chargerL, chargerR

    def distanceTo(self, obj):
        xx, yy = obj.getLocation()
        return math.sqrt(math.pow(self.x - xx, 2) + math.pow(self.y - yy, 2))

    def update(self, canvas, passiveObjects, dt):
        from others import Charger
        self.battery -= 1
        for rr in passiveObjects:
            if isinstance(rr, Charger) and self.distanceTo(rr) < 80:
                self.battery += 30
        if self.battery <= 0:
            self.battery = 0
        self.move(canvas, dt)

    def draw(self, canvas):
        self.cameraPositions = []
        for pos in range(20, -21, -5):
            self.cameraPositions.append(
                ((self.x + pos * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta), \
                 (self.y - pos * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta)))
        for xy in self.cameraPositions:
            canvas.create_oval(xy[0] - 2, xy[1] - 2, xy[0] + 2, xy[1] + 2, fill="purple1", tags=self.name)
        for xy in self.cameraPositions:
            canvas.create_line(xy[0], xy[1], xy[0] + 400 * math.cos(self.theta), xy[1] + 400 * math.sin(self.theta),
                               fill="light grey", tags=self.name)

        points = [(self.x + 30 * math.sin(self.theta)) - 30 * math.sin((math.pi / 2.0) - self.theta), \
                  (self.y - 30 * math.cos(self.theta)) - 30 * math.cos((math.pi / 2.0) - self.theta), \
                  (self.x - 30 * math.sin(self.theta)) - 30 * math.sin((math.pi / 2.0) - self.theta), \
                  (self.y + 30 * math.cos(self.theta)) - 30 * math.cos((math.pi / 2.0) - self.theta), \
                  (self.x - 30 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta), \
                  (self.y + 30 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta), \
                  (self.x + 30 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta), \
                  (self.y - 30 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta) \
                  ]
        canvas.create_polygon(points, fill="blue", tags=self.name)

        self.sensorPositions = [(self.x + 20 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta), \
                                (self.y - 20 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta), \
                                (self.x - 20 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta), \
                                (self.y + 20 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta) \
                                ]

        centre1PosX = self.x
        centre1PosY = self.y
        canvas.create_oval(centre1PosX - 16, centre1PosY - 16, \
                           centre1PosX + 16, centre1PosY + 16, \
                           fill="gold", tags=self.name)
        canvas.create_text(self.x, self.y, text=str(self.battery), tags=self.name)

        wheel1PosX = self.x - 30 * math.sin(self.theta)
        wheel1PosY = self.y + 30 * math.cos(self.theta)
        canvas.create_oval(wheel1PosX - 3, wheel1PosY - 3, \
                           wheel1PosX + 3, wheel1PosY + 3, \
                           fill="red", tags=self.name)

        wheel2PosX = self.x + 30 * math.sin(self.theta)
        wheel2PosY = self.y - 30 * math.cos(self.theta)
        canvas.create_oval(wheel2PosX - 3, wheel2PosY - 3, \
                           wheel2PosX + 3, wheel2PosY + 3, \
                           fill="green", tags=self.name)

        sensor1PosX = self.sensorPositions[0]
        sensor1PosY = self.sensorPositions[1]
        sensor2PosX = self.sensorPositions[2]
        sensor2PosY = self.sensorPositions[3]
        canvas.create_oval(sensor1PosX - 3, sensor1PosY - 3, \
                           sensor1PosX + 3, sensor1PosY + 3, \
                           fill="yellow", tags=self.name)
        canvas.create_oval(sensor2PosX - 3, sensor2PosY - 3, \
                           sensor2PosX + 3, sensor2PosY + 3, \
                           fill="yellow", tags=self.name)

    def move(self, canvas, dt):
        if self.battery == 0:
            self.sl = 0
            self.sr = 0

        # 直线运动
        if self.sl == self.sr:
            newX = self.x + self.sr * math.cos(self.theta) * dt
            newY = self.y + self.sr * math.sin(self.theta) * dt
            newTheta = self.theta
        else:
            R = (self.ll / 2.0) * ((self.sr + self.sl) / (self.sl - self.sr))
            omega = (self.sl - self.sr) / self.ll
            ICCx = self.x - R * math.sin(self.theta)
            ICCy = self.y + R * math.cos(self.theta)

            # 使用矩阵计算新位置
            m = np.matrix([[math.cos(omega * dt), -math.sin(omega * dt), 0],
                           [math.sin(omega * dt), math.cos(omega * dt), 0],
                           [0, 0, 1]])
            v1 = np.matrix([[self.x - ICCx], [self.y - ICCy], [self.theta]])
            v2 = np.matrix([[ICCx], [ICCy], [omega * dt]])
            newv = np.add(np.dot(m, v1), v2)

            newX = newv.item(0)
            newY = newv.item(1)
            newTheta = newv.item(2)

        # 规范化角度
        newTheta = newTheta % (2.0 * math.pi)

        # 边界检测和处理
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

        # 如果发生碰撞，减速
        if collided:
            self.sl *= 0.5
            self.sr *= 0.5
            print(f"{self.name} hit wall.")

        # 更新机器人的位置和角度
        self.x = newX
        self.y = newY
        self.theta = newTheta

        # 删除画布上原来的机器人，并在新位置绘制机器人
        canvas.delete(self.name)
        self.draw(canvas)

    def collectCatFur(self, canvas, passiveObjects, count):
        """清理猫毛并记录数量"""
        from fur import Fur  # 假设猫毛是 Fur 类的实例
        toDelete = []
        for idx, fur in enumerate(passiveObjects):
            if isinstance(fur, Fur) and self.distanceTo(fur) < 30:
                canvas.delete(fur.tag)
                toDelete.append(idx)
                count.itemCollected(canvas)
        # 删除收集的猫毛
        for ii in sorted(toDelete, reverse=True):
            del passiveObjects[ii]
        return passiveObjects

    def collectDirt(self, canvas, passiveObjects, count):
        from others import Dirt
        toDelete = []
        for idx, rr in enumerate(passiveObjects):
            if isinstance(rr, Dirt):
                if self.distanceTo(rr) < 30:
                    canvas.delete(rr.name)
                    toDelete.append(idx)
                    count.itemCollected(canvas)
        for ii in sorted(toDelete, reverse=True):
            del passiveObjects[ii]
        return passiveObjects

    def collision(self, agents):
        from cat import Cat
        collision = False
        for rr in agents:
            if isinstance(rr, Cat):
                if self.distanceTo(rr) < 50.0:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load("cat.mp3")
                    pygame.mixer.music.play()
                    collision = True
                    rr.jump()
        return collision
