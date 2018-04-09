import re
from sys import argv
#script, filename = arg
input = open('text.txt', 'r+')
output = open('test.txt', 'w+')
txt = input.read()
txt = (re.sub(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z][a-z]\.)(?<![A-Z][a-z]\.)(?!')(?<=\.|\?)", "<\\s>", txt))
# print "txt1-------------------------------- \n" + txt1
txt = (re.sub(r"(?<=\.'|\!'|\?')\n", "<\\s>\\n", txt))
# print "txt2------------------------\n" + txt2
txt = (re.sub(r"(?<=(<\\s>))(?!\Z)(\s+)","\n<s>",txt))
txt = '<s>' + txt[:-4]
array = re.findall(r"(?<=<s>)((.|\n)+?)(?=<\\s>)",txt)
array1 = [i[0] for i in array]
array2 = []
# filtering out newline character
for temp in array1:
    temp = re.sub(r"\n", " ", temp) 
    temp = re.sub(r"\s\s", "\\n", temp)
    array2.append(temp)

for temp in array2:
  output.write('<s>' + temp + '</s>\n')
input.close()
output.close()
