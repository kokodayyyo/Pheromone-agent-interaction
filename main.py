import tkinter as tk
from others import initialise, createObjects, moveIt


def main():
    window = tk.Tk()
    canvas = initialise(window)
    agents, passiveObjects, count = createObjects(canvas, noOfBots=2, noOfLights=0, amountOfDirt=300, noOfCats=5)
    moveIt(canvas, agents, passiveObjects, count, 0)
    window.mainloop()


main()
