import re
input = open('text.txt', 'r+')
txt = input.read() 
output= open("test.txt","w+")
output.write(re.sub(r"\B\'|\'\B", "\"", txt))
input.close()
output.close()
