import os, time
import skfuzzy as fuzz, numpy as np

X = np.arange(0,1.0001,0.0001)
X = np.asarray([np.round(i,4) for i in X])

FS1 = ['unimportant','average','important']
FS2 = ['L','M','H']

lab1 = dict([(i,j) for j,i in enumerate(FS1)])
lab2 = dict([(i,j) for i,j in enumerate(FS2)])

F1L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F1M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F1H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])

F2L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F2M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F2H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])

F3L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F3M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F3H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])

F4L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F4M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F4H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])

F5L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F5M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F5H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])

F6L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F6M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F6H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])

F7L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F7M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F7H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])

F8L = fuzz.trapmf(X,[-1, -1, 0, 0.2])
F8M = fuzz.trapmf(X,[0.05, 0.18, 0.4, 0.6])
F8H = fuzz.trapmf(X,[0.4, 0.6, 1, 1.1])


NS = open('Ns.txt').read().split()
NS = np.int32(NS)

Unimportant = fuzz.trapmf(X,[-1, -1, 0, 0.2])
Average     = fuzz.trapmf(X,[0,0.19,0.4, 0.55])
Important   = fuzz.trapmf(X,[0.4,0.6,1, 1.1])

tri = dict([('unimportant',[0,  0, 0,0.2]),
            ('average',    [0,0.19,0.4, 0.55]),
            ('important',  [0.4,0.6,1, 1])])


# opening frules files and making it a list of tuples of the form ([l, l, ...], important)
frules = [i.strip().split() for i in open('frules.txt').readlines()]
frules = [(i[:-1],i[-1].lower()) for i in frules]

index = dict([(j,i) for i,j in enumerate(X)])

#path_inp1 = './2.feature/'
path_inp1 = './4.weighted_feature/'
path_inp2 = './0.dataset raw/'
path_out1 = './5.scored/'
path_out2 = './6.summarized/'

for enum,file in enumerate(os.listdir(path_inp1)):
    start = time.time()
    print(file,end=' ')

    f = open(path_inp1+file).read()

# data will contain all weighted score as a matrix, label is a list of all sentence labels
    data,label = [],[]
    for row in f.splitlines():
        temp = []
        label.append(row.split()[0])
        for col in row.split()[1:]:
            temp.append(float(col))
        data.append(temp)
    data = np.asarray(data)

    score = []
    for x in data:

        FS,FV = ['','',''],[0,0,0]

        for R,fs in frules:
            a,m,be,b = tri[fs]

            c = []
            for j,r in enumerate(R):
                val = np.round(x[j],4)
                c.append(eval('F'+str(j+1)+r)[index[val]])
            minc = min(c)
        
            if FS[lab1[fs]] == '':
                FS[lab1[fs]] = fs
                FV[lab1[fs]] = minc
            else:
                if minc > FV[lab1[fs]]:
                    FV[lab1[fs]] = minc
        
        #print(FV)

        U = [i if i <= FV[0] else FV[0] for i in Unimportant]
        A = [i if i <= FV[1] else FV[1] for i in Average]
        I = [i if i <= FV[2] else FV[2] for i in Important]
    
        mfx = np.max(np.vstack((U,A,I)),0)
        score.append(fuzz.defuzz(X,mfx,'centroid'))

    N = NS[enum]
    print(N)
    N_best = [label[i].split('_')[1] for i in np.argsort(score)[::-1][:N]]
    print("best")
    print(N_best)
    N_best = sorted(N_best)

    numb = file.split('.')[0] #Get file name
    f_in   = open(path_inp2+str(int(numb))+'.txt').read()
    f_out1 = open(path_out1+file,'w')
    f_out2 = open(path_out2+file,'w')

    v = 0
    for i,sent in enumerate(f_in.splitlines()):
        numb = '0'*(2-len(str(i-1)))+str(i-1)
        strg = 'sent_'+numb
        if i == 0:
            f_out1.write(sent+'\n')
            f_out2.write(sent+'\n')
            continue
        
        if strg in label:
            vals = str(np.round(score[v],5))
            v += 1
        else:
            vals = '-'

        if numb in N_best:
            f_out2.write(sent+'\n')

        #formatting scores
        vals = vals+' '*(8-len(vals))
        sent = sent[:100]+'...' if len(sent) > 100 else sent
        f_out1.write(strg+' | '+vals+'| '+sent+'\n')
    f_out1.close()
    f_out2.close()

    with open('./3.reference/001.txt') as f:
        num_lines_ref = sum(1 for _ in f)

    with open(path_out2 + '001.txt') as f:
        num_lines_sys = sum(1 for _ in f)
    print(num_lines_sys)
    print(num_lines_ref)

    while num_lines_sys > num_lines_ref:
        print(num_lines_sys - num_lines_ref)
        fd = open(path_out2 + '001.txt', "r")
        d = fd.read()
        fd.close()
        m = d.split("\n")
        s = "\n".join(m[:-1])
        fd = open(path_out2 + '001.txt', "w+")
        for i in range(len(s)):
            fd.write(s[i])
        fd.close()
        with open(path_out2 + '001.txt') as f:
            num_lines_sys = sum(1 for _ in f)
    
    print('time:',time.time()-start)
