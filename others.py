from playsound import playsound
from PIL import Image, ImageTk
import tkinter as tk
import random
import math
import numpy as np
import time
import sys

class Lamp():
    def __init__(self, namep):
        self.centreX = random.randint(100, 900)
        self.centreY = random.randint(100, 900)
        self.name = namep

    def draw(self, canvas):
        body = canvas.create_oval(self.centreX - 10, self.centreY - 10, \
                                  self.centreX + 10, self.centreY + 10, \
                                  fill="yellow", tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class Charger():
    def __init__(self, namep):
        self.centreX = random.randint(100, 900)
        self.centreY = random.randint(100, 900)
        self.name = namep

    def draw(self, canvas):
        body = canvas.create_oval(self.centreX - 10, self.centreY - 10, \
                                  self.centreX + 10, self.centreY + 10, \
                                  fill="red", tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class WiFiHub:
    def __init__(self, namep, xp, yp):
        self.centreX = xp
        self.centreY = yp
        self.name = namep

    def draw(self, canvas):
        body = canvas.create_oval(self.centreX - 10, self.centreY - 10, \
                                  self.centreX + 10, self.centreY + 10, \
                                  fill="purple", tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class Dirt:
    def __init__(self, namep):
        self.centreX = random.randint(100, 900)
        self.centreY = random.randint(100, 900)
        self.name = namep

    def draw(self, canvas):
        body = canvas.create_oval(self.centreX - 1, self.centreY - 1, \
                                  self.centreX + 1, self.centreY + 1, \
                                  fill="grey", tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class Counter:
    def __init__(self):
        self.dirtCollected = 0

    def itemCollected(self, canvas):
        self.dirtCollected += 1
        canvas.delete("dirtCount")
        canvas.create_text(50, 50, anchor="w", \
                           text="Dirt collected: " + str(self.dirtCollected), \
                           tags="dirtCount")


def initialise(window):
    window.resizable(False, False)
    canvas = tk.Canvas(window, width=1000, height=1000)
    canvas.pack()
    return canvas


def buttonClicked(x, y, agents):
    from bot import Bot
    for rr in agents:
        if isinstance(rr, Bot):
            rr.x = x
            rr.y = y


def createObjects(canvas, noOfBots=2, noOfLights=2, amountOfDirt=300, noOfCats=5):
    from bot import Bot,Brain
    from cat import Cat, CatNest
    agents = []
    passiveObjects = []
    cat_nest = CatNest

    for i in range(0, noOfBots):
        bot = Bot("Bot" + str(i), canvas)
        brain = Brain(bot)
        bot.setBrain(brain)
        agents.append(bot)
        bot.draw(canvas)

    for i in range(0, noOfCats):
        cat = Cat("Cat" + str(i), canvas, cat_nest)
        agents.append(cat)
        cat.draw(canvas)

    for i in range(0, noOfLights):
        lamp = Lamp("Lamp" + str(i))
        passiveObjects.append(lamp)
        lamp.draw(canvas)

    charger = Charger("Charger" + str(i))
    passiveObjects.append(charger)
    charger.draw(canvas)

    hub1 = WiFiHub("Hub1", 950, 50)
    passiveObjects.append(hub1)
    hub1.draw(canvas)
    hub2 = WiFiHub("Hub2", 50, 500)
    passiveObjects.append(hub2)
    hub2.draw(canvas)

    for i in range(0, amountOfDirt):
        dirt = Dirt("Dirt" + str(i))
        passiveObjects.append(dirt)
        dirt.draw(canvas)

    count = Counter()

    canvas.bind("<Button-1>", lambda event: buttonClicked(event.x, event.y, agents))

    return agents, passiveObjects, count


def moveIt(canvas, agents, passiveObjects, count, moves):
    from bot import Bot
    from cat import Cat  # 确保导入 Cat 类

    # 清除之前的信息素可视化
    canvas.delete("pheromone")

    for rr in agents:
        if isinstance(rr, Cat):
            passiveObjects = rr.thinkAndAct(agents, passiveObjects, canvas)  # Cat 的行为
            rr.update(canvas, passiveObjects, 1.0)
        else:
            rr.thinkAndAct(agents, passiveObjects, canvas)
            rr.update(canvas, passiveObjects, 1.0)
            if isinstance(rr, Bot):
                passiveObjects = rr.collectDirt(canvas, passiveObjects, count)  # 收集灰尘
                passiveObjects = rr.collectCatFur(canvas, passiveObjects, count)  # 收集猫毛
                rr.brain.draw_pheromones(canvas)

    canvas.after(20, moveIt, canvas, agents, passiveObjects, count, moves)
