import argparse
from string import printable
import re
from joblib import Parallel, delayed
from multiprocessing import Pool 
import multiprocessing as mp
import asyncio
import time
#import os
#os.system("taskset -p 0xff %d" % os.getpid())
#def background(f):
#    def wrapped(*args, **kwargs):
#        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#
#    return wrapped
#    
#@background 
def processInner(target_lines, originalSrc, currSourceHypLine):
   with open(args.target, 'a', encoding="utf-8") as w:
     for currentIndexTgt, originalSrcLine in enumerate(originalSrc):
          if currSourceHypLine == originalSrcLine.lower().replace(' ', '').strip():
              w.write(target_lines[currentIndexTgt])
              w.write("\n")
              del originalSrc[currentIndexTgt]
              return
     print("not found, removing from hyp file")        
   return currentIndexTgt        

       
               
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--originalSrc', '--originalSrc', type=str, help='Original Src file')
    parser.add_argument('--originalTgt', '--originalTgt', type=str, help='Original Tgt file')
    parser.add_argument('--sourceHyp', '--srcHyp', type=str, help='File with each line-by-line model outputs')
    parser.add_argument('--target', '--tgt', type=str, help='Output file name')
    args = parser.parse_args()
    print("IN CREATE REF FILE")
    sourceHyp = []
    with open(args.sourceHyp, 'r', encoding="utf-8") as f:
        for line in f:
            sourceHyp.append(line.strip())
            


    toRemoveFromSourceHyp = []
    with open(args.target, 'w+', encoding="utf-8") as w, open(args.originalSrc, 'r', encoding="utf-8") as r, open(args.originalTgt, 'r', encoding="utf-8") as originalTgt:
        currentIndex = 0
        lengthOfSourceHyp = len(sourceHyp)
        originalSrc = []
        target_lines = []
        for line in r:
            originalSrc.append(line.strip())

        
        for line in originalTgt:
            target_lines.append(line.strip())
            
        print(target_lines[0])
        print("LENGTH OF SOURCE HYP: " + str(lengthOfSourceHyp))
        print("LENGTH OF ORIGINAL SRC: " + str(len(originalSrc)))
        childProcesses = []
        cpu_count = mp.cpu_count()
        print(cpu_count)
        pool = mp.Pool(cpu_count)
        for currentIndexTgt, sourceHypLine in enumerate(sourceHyp):
            #print(currentIndexTgt)
            if (currentIndexTgt % 100000 == 0):
              print(currentIndexTgt)
              
            currSourceHypLine = sourceHypLine.lower().replace(' ', '').strip()
            childProcesses.append((target_lines, originalSrc, currSourceHypLine))

        print("about to do: ")
        print(len(childProcesses))
        print("subprocesses")   
        toRemoveFromSourceHyp = pool.starmap(processInner, childProcesses)
        print('Waiting for the process...') 
        pool.close()
        pool.join()        
        
        
        #process(sourceHyp, toRemoveFromSourceHyp, originalSrc, w, target_lines)    
        print("Actually finished... waiting 20 seconds")
        time.sleep(20)
        print(toRemoveFromSourceHyp)

    print("should be over now??") 
    with open(args.sourceHyp, "w", encoding="utf-8") as f:
        for i, line in enumerate(sourceHyp):
            if i not in toRemoveFromSourceHyp:
                f.write(line)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    
#    
#    
#import argparse
#from string import printable
#import re
#from joblib import Parallel, delayed
#import asyncio
#import time
#import os
#os.system("taskset -p 0xff %d" % os.getpid())
#def background(f):
#    def wrapped(*args, **kwargs):
#        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#
#    return wrapped
#    
#@background 
#def processInner(target_lines, originalSrc, w, toRemoveFromSourceHyp, currSourceHypLine):
#   for currentIndexTgt, originalSrcLine in enumerate(originalSrc):
#        if currSourceHypLine == originalSrcLine.lower().replace(' ', '').strip():
#            w.write(target_lines[currentIndexTgt])
#            w.write("\n")
#            del originalSrc[currentIndexTgt]
#            return True
#   print("not found, removing from hyp file")         
#   toRemoveFromSourceHyp.append(currentIndexTgt)            
#
#       
#def process(sourceHyp, toRemoveFromSourceHyp, originalSrc, w, target_lines):
#    for currentIndexTgt, sourceHypLine in enumerate(sourceHyp):
#        #print(currentIndexTgt)
#        if (currentIndexTgt % 100000 == 0):
#          print(currentIndexTgt)
#          
#        currSourceHypLine = sourceHypLine.lower().replace(' ', '').strip()
#        processInner(target_lines, originalSrc, w, toRemoveFromSourceHyp, currSourceHypLine)
#        
#            
#               
#    
#    
#if __name__ == '__main__':
#
#    parser = argparse.ArgumentParser()
#
#    parser.add_argument('--originalSrc', '--originalSrc', type=str, help='Original Src file')
#    parser.add_argument('--originalTgt', '--originalTgt', type=str, help='Original Tgt file')
#    parser.add_argument('--sourceHyp', '--srcHyp', type=str, help='File with each line-by-line model outputs')
#    parser.add_argument('--target', '--tgt', type=str, help='Output file name')
#    args = parser.parse_args()
#    print("IN CREATE REF FILE")
#    sourceHyp = []
#    with open(args.sourceHyp, 'r', encoding="utf-8") as f:
#        for line in f:
#            sourceHyp.append(line.strip())
#            
#
#
#    toRemoveFromSourceHyp = []
#    with open(args.target, 'w+', encoding="utf-8") as w, open(args.originalSrc, 'r', encoding="utf-8") as r, open(args.originalTgt, 'r', encoding="utf-8") as originalTgt:
#        currentIndex = 0
#        lengthOfSourceHyp = len(sourceHyp)
#        originalSrc = []
#        target_lines = []
#        for line in r:
#            originalSrc.append(line.strip())
#
#        
#        for line in originalTgt:
#            target_lines.append(line.strip())
#            
#        print(target_lines[0])
#        print("LENGTH OF SOURCE HYP: " + str(lengthOfSourceHyp))
#        print("LENGTH OF ORIGINAL SRC: " + str(len(originalSrc)))
#        process(sourceHyp, toRemoveFromSourceHyp, originalSrc, w, target_lines)    
#        print("Actually finished... waiting 20 seconds")
#        time.sleep(20)
#        print(toRemoveFromSourceHyp)
#
#    print("should be over now??") 

