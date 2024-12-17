

safe_counter = 0

input = open("input.txt", "r")

for i in input.readlines():
    A = i.split()
    A = list(map(int, A))
    if all(j < k and k < j + 4 for j,k in zip(A, A[1:])) or all(j > k and k > j - 4 for j,k in zip(A, A[1:])):
        safe_counter += 1
        continue
    
    for m in range(len(A)):
        B = A[:m] + A[m+1:]
        if all(j < k and k < j + 4 for j,k in zip(B, B[1:])) or all(j > k and k > j - 4 for j,k in zip(B, B[1:])):
            safe_counter += 1
            break


print(safe_counter)





            