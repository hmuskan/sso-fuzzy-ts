import xml.etree.ElementTree as ET
import os
from nltk.tokenize import sent_tokenize

# read files one by one
# parse xml
# get the text, split linewise and add to final var
# write final variable
path_inp1 = './0.data/docs/'
data = []
for enum,file in enumerate(os.listdir(path_inp1)):
    print(file,end=' ')
    tree = ET.parse(path_inp1+file)
    root = tree.getroot()
    text = root.findall('TEXT')[0].text.replace("\n", " ").strip()
    data.append(sent_tokenize(text))

#print(data)
f = open('./0.dataset raw/1.txt','w')
fdata = ""
for d in data:
    fdata += ("\n".join(d)) + "\n"
f.write(fdata)
f.close()

path_inp2 = './0.data/summary/'
data = []
for enum,file in enumerate(os.listdir(path_inp2)):
    print(file,end=' ')
    tree = ET.parse(path_inp2+file)
    root = tree.getroot()
    text = root.text.replace("\n", " ").strip()
    #print(text)
    data.append(sent_tokenize(text))

num = len(data[0])
f = open('./Ns.txt','w')
f.write(str(num))
f.close()
f = open('./3.reference/001.txt','w')
fdata = ""
for d in data:
    fdata += ("\n".join(d)) + "\n"
f.write(fdata)
f.close()


