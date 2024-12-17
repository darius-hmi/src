
import re

with open('input.txt', 'r') as file:
    txt = file.read().replace('\n', '')

    multiplications = re.findall(r'don\'t\(\)|do\(\)|mul\(\d{1,3},\d{1,3}\)', txt)

    Total = 0
    active = True
    for i in multiplications:
        if i == 'do()':
            active = True
        elif i == "don't()":
            active = False
        elif i.startswith("mul(") and active:
            numbers = i.strip('mul()')
            a = int(numbers.split(',')[0])
            b = int(numbers.split(',')[1])
            mult = a*b
            Total = Total + mult


        
    
    print(Total)

    