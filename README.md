信息素存储在一个二维网格（pheromone_grid）中，每个网格单元存储了该位置的信息素强度和类型。网格的大小由cell_size决定。
机器人通过感知周围网格单元中的信息素来调整行为。感知时会考虑信息素的类型和强度，并根据其衰减率计算当前的有效强度。
梯度：通过信息素强度计算梯度，依据情况向着梯度升高或者降低的方向前进
（CHASE）当机器人检测到目标（如猫）的位置时，会在目标附近释放追逐信息素。机器人会根据感知到的追逐信息素的梯度强度和方向，调整自身的路径规划。信息素强度较高的区域被认为是目标更接近的地方，机器人会优先向这些区域移动。
（EXPLORE）机器人在移动过程中，会定期在自身位置释放探索信息素，以标记已探索的区域。机器人会优先选择探索信息素梯度强度较低的区域进行探索，以避免重复探索已标记的区域。
（ESCAPE）当猫感知到潜在的威胁（如对方靠近），会在自身周围释放逃跑信息素。机器人会根据感知到的逃跑信息素调整路径，追踪信息素强度较高的区域，即跟踪猫走过的路径。
（DECOY）当目标对象希望误导追踪者（如机器人）时，会在自身周围或特定位置释放诱饵信息素。机器人会根据感知到的诱饵信息素调整路径，误以为诱饵信息素标记的位置为目标位置，从而被误导。（加入brain类中计算逃跑信息素的梯度里）。
