import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pydicom
from pydicom.data import get_testdata_files
import os, shutil

idx = 0
absPath = 'C:\\Users\\daska\\Documents\\AIN\\MAGISTER\\Diplomovka\\Data\\'
absPathFrom = 'C:\\Users\\daska\\Documents\\AIN\\MAGISTER\\Diplomovka\\'

def save(co, kam):
    #print(absPathFrom + co)
    #print(absPath + kam)
    shutil.copyfile(absPathFrom + co, absPath + kam + '.dcm')


axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
ax = plt.axes([0.001, 0.15, 1, 1])
btrue = Button(axnext, 'JE')
#btrue.on_clicked(lambda x: save('Data\prostate_true', x))
bfalse = Button(axprev, 'NIE je')
#bfalse.on_clicked(lambda x: save('Data\prostate_false', x))
plt.show(block=False)

def traverse(path, our=False):
    for fn in os.listdir(path):
        fpath = os.path.join(path, fn)
        #print(fpath)
        if os.path.isfile(fpath):
            if our:
                draw(fpath)
        else:
            if 'T2' in fn and 'ax' in fn:
                traverse(fpath, True)
            else:
                traverse(fpath, False)

def draw(file):    
    ds=pydicom.read_file(file)
    #print(f"Image size.......: {ds.Rows} x {ds.Columns}")
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    btrue.on_clicked(lambda x: save(file,'prostata_true\\' + "file" + str(idx)))
    bfalse.on_clicked(lambda x: save(file, 'prostata_false\\' + "file" + str(idx)))
    global idx
    idx = idx + 1
    plt.draw()
    plt.waitforbuttonpress()
    

traverse('CIA_data')


##filename = get_testdata_files('test.dcm')
##print(filename)
##ds = pydicom.dcmread(filename)
##filePath = '1-08.dcm'
##ds=pydicom.read_file(filePath)
##plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
##plt.show()

