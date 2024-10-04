import os
import pandas as pd
import shutil
from PIL import Image

"""


train_df=pd.read_csv('fakeproduct/test/_annotations.csv')
fakes_index=train_df.index[train_df['class']=='fake']
for fake in fakes_index:
    print(fake)
    filename=train_df.iloc[fake]['filename']
    print(filename)
    dir='fakeproduct/test/'+ str(filename)
    print(dir)

    shutil.copyfile(dir, 'pytorchdata/test/fake/'+str(filename))


real_index=train_df.index[train_df['class']=='real']
for real in real_index:
    print(fake)
    filename=train_df.iloc[real]['filename']
    print(filename)
    dir='fakeproduct/test/'+ str(filename)
    print(dir)

    shutil.copyfile(dir, 'pytorchdata/test/real/'+str(filename))

    """
paths=['pytorchdata/test/real','pytorchdata/test/fake','pytorchdata/train/real','pytorchdata/train/fake']
for p in paths:
    for file in os.listdir(p):
        
        
        
        image = Image.open(str(p)+'/'+str(file))
        new_image = image.resize((256, 256))
        chunks=p.split('pytorchdata/')
        new_image.save('pytorchResized/'+str(chunks[1])+'/'+str(file))
        