import sys
import time

# Get file to run
filename = sys.argv[1]
outname = filename[:-2] + "figtime"

# Run & time it
start = time.time()
exec(open(filename).read())
seconds = time.time() - start

# Output the time
with open(outname, "w") as f:
    print(time.strftime("%Mm%Ss", time.gmtime(seconds)), file=f)
