import os
import re 
import pandas as pd
import xml.etree.ElementTree as ET

# path to complete scanned file -- change here for testing different filetypes
file = './ocr_work/combined-txt-full-tif-cropped.txt' 

with open(file) as file:
    lines = file.readlines()

text = []
for line in lines:
    line = line.lower()
    line = line.replace("\n", " ")
    line = line.replace("  ", " ")
    line = line.replace("  ", " ")
    line = line.replace('"', " ")
    line = line.replace('or.', 'dr.')
    line = re.sub('[c,t,s,o,e,n,m,e,a,r,u,i,b,w,p,r,v,d]{18,}', "", line)
    if(bool(re.search("[A-Za-z0-9]", line))==True):
        text.append(line)


class DT():
    def __init__(self, Name, President):
        self.name = Name
        self.president = President

    def print(self):
        print(self.name + " --- " + self.president)


def grab(ws):
    terms = ['accred', 'accrad']
    ixs = []

    for term in terms:
        ixs+=get_ix(ws, term)
    
    pgs = []
    for ix in ixs:
        length = 6
        for i in range(length):
            if ix+length in ixs:
                ixs.remove(ix+length)

    for i in range(len(ixs)):
        if(len(ixs)>i+1):
            pgs.append(ws[ixs[i]:ixs[i+1]])

    return pgs


def get_ix(ws, term):
    ix = []
    for i in range(len(ws)):
        if(term in ws[i]):
            ix.append(i)
    return ix

def get_pres(data):
    ignore = ['page break', 'community', 'librariin', 'cellag', 'coluga', 'offica', 'chancelior', 'aisistant', 'theological', 'collage', 'computing', 'highest offering', "master's", ' materials ', 'fongr. dist', 'congr dist', 'congr. dist', ' assoc dir ', '*u ', 'ubrarian', 'accraditation', 'fiscal', 'association', 'bachelor', 'institute',  'village of','division of', 'inst of', ' resource ', 'u off ', 'religion', 'hussar', 'enrollment', 'u of ', 'vice', 'accredkation', 'accredkatio', 'counselling', 'telephone', 'agrarian', 'register', 'chairman', 'fice rode', 'univonity', 'university', 'list.', 'dist.', 'supervisor of', 'comptroller', 'school of', 'acad plan',  'alumnae', 'activity', 'activities', 'coordinator', 'education', 'library', 'accreditition', 'provost', 'program', 'secretary', 'financial', 'control', 'librerian', 'ragistrar', 'computer', 'planning', 'institutional', 'research', 'controller', 'vocational', 'technical', 'placement', 'counselor', 'chancellor', 'quarter', 'coordinate', 'college', 'treasurer', 'bursar', 'acadtmic', 'trimester', 'diractor', 'public', 'relations', 'assistant', 'vice', 'registrar', 'semester', 'accreditation', 'academic', 'business', 'manager', 'reigstrar', 'director', 'admission', 'librarian', 'congr.', 'dist.', 'office']
    # CASE 1 ::: president name straightup
    for line in data:
        if ('pres' in line or 'resident' in line) and 'vice' not in line:
            check = line.replace('president', '')
            check = re.sub('[.]*', '', check)
            if(bool(re.search("[A-Za-z]", check))==True):
                return line
    
    # CASE 2 ::: return any name in line directly after accred ---> 
    for i in range(len(data)):
        if ('accred' in data[i]) and len(data)>i+1:
            return data[i+1]
            
    # CASE 3 ::: president name some line after accred --->
    for i in range(len(data)):
        if ('accred' in data[i]) and len(data)>i+1:
            for j in range(len(data)-(i+1)):
                check = data[i+j+1]
                tst = check.replace('president', '')
                tst1 = check.replace('president and dean', '')
                tst2 = check.replace('president and mean', '')
                tst3 = check.replace('acting president', '')
                tst4 = check.replace('old', '')
                tst5 = check.replace('superintendent', '')
                if(bool(re.search("[A-Za-z]", tst))==True) and (bool(re.search("[A-Za-z]", tst1))==True) and (bool(re.search("[A-Za-z]", tst2))==True) and (bool(re.search("[A-Za-z]", tst3))==True) and (bool(re.search("[A-Za-z]", tst4))==True) and (bool(re.search("[A-Za-z]", tst5))==True):
                    if check_present(check, ignore) == False:
                        return check

def get_school(data):
    terms = [' offering', 'coll', 'preparatory', 'seminary', 'saminory', 'collwgw', 'cellwgw', 'cellogo', ' of ', 'institut', 'office', ' u ', 'state', 'uni', 'univ', 'univmitity', 'univofsity', 'univeriify', 'univerify', 'univonity', 'univanity', 'coluga', 'inst', ' jc', 'campus', 'colloga', 'university', 'college', 'u of', 'collage', 'suny', 'supv', 'tchrs', 'teach', 'technl', 'tedc', 'test', 'text', 'theo', 'theol', 'thos', 'trade', 'union', 'undergrad', 'undergraduate', 'univ', 'uni', '*']
    ignore = ['or. ', 'dr. ', 'financial', 'officer', 'accreditation', ', ', 'teleph', 'student affairs', 'registrar', 'director', 'donation', ' list ', 'dist.', 'list.', 'mean', 'mean of', 'director of', 'dir of', 'program—', 'assistant', 'assnt ', 'dr. ' 'adminv', ' dist ', 'street', 'congr.', 'dean of', 'avenue', 'highway', 'conr.', 'mean of']
    length = 5
    #checks for lines directly before the telephone number to find college name
    for i in range(len(data)):
        if ("tele" in data[i]) or (bool(re.search("\([0-9][0-9][0-9]\)", data[i]))==True):
            if (i<length):
                length = i
            for j in range(length):
                curr = data[i-j]
                if check_present(curr, ignore) == False:
                    for term in terms:
                        if term in curr:
                            return curr

# check overlapping elts in a list
def check_present(str, ls):
    for l in ls:
        if l in str:
            return True
    return False


###
###
# scraping proccess ::::::
###
###


chunks = grab(text)
dts = []
p_count = 0
s_count = 0
for i in range(len(chunks)):
    if len(chunks)>i+1:
        txt = chunks[i]
        next_txt = chunks[i+1]
        p = get_pres(next_txt)
        s = get_school(txt)
        if s!=None:
            s_count+=1
        if p!=None:
            p_count+=1
        dt = DT(s, p)
        dts.append(dt)


ps = []
ss = []
for dt in dts:
    ps.append(dt.president)
    ss.append(dt.name)

#counts for sanity check ---
print(len(chunks))
print(s_count)
print(p_count)

#save to cvs file
df = pd.DataFrame({'president names': ps,'schools': ss})
df.to_csv('results.csv', index=False)