


input = open("input.txt", "r")

A = []
B = []
D = []
new_D = []


for i in input.readlines():
    a = i.split("   ")[0]
    b = i.split("   ")[1].rstrip("\n")
    
    A.append(int(a))
    B.append(int(b))


A.sort()
B.sort()


for i, j in zip(A, B):
    d = abs(i - j)
    D.append(d)

total = sum(D)

print(total)



for i in A:
    d = i * B.count(i)
    new_D.append(d)


print(sum(new_D))