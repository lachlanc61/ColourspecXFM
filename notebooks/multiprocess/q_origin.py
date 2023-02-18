from multiprocessing import Process, Queue

fn="/home/lachlan/CODEBASE/ReadoutXFM/notebooks/multiprocess/temp.txt"
#fn="/home/lachlan/CODEBASE/ReadoutXFM/data/ts2.GeoPIXE"

"""
def getfile(fn, out):
    with open(fn) as file:
        for line in file:
            print(line)

out=""

getfile(fn, out)
"""

def getfile(fn, out):
    with open(fn) as file:
        for line in file:
            out.put(line)

if __name__ == '__main__':
    file1 = Queue()
    p1 = Process(target=getfile, args=("multiprocess.py", file1))
    p1.start()
    file1.get() #returns lines of the file
    p1.join()
