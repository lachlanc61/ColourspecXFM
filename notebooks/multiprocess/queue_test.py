from multiprocessing import Process, Queue


fn="/home/lachlan/CODEBASE/ReadoutXFM/data/ts2.GeoPIXE"
"""
with open(fn, mode='rb') as file:
    data=file.read(2000)
    #print(len(data))
    print(data[1498:2000])
"""

def getfile(fn, out):
    with open(fn, mode='rb') as file:
        data=file.read(2000)
        out.put(data)

if __name__ == '__main__':
    file1 = Queue()
    p1 = Process(target=getfile, args=(fn, file1))
    p1.start()
    file1.get() #returns data from file
    p1.join()





"""
from multiprocessing import Process, Queue

def getfile(fn, out):
    with open(fn) as file:
        for line in file:
            out.put(line)

if __name__ == '__main__':
    file1 = Queue()
    p1 = Process(target=getfile, args=("100.txt", file1))
    p1.start()
    file1.get() //returns lines of the file
    p1.join()
"""