def gen():
    i = 0
    while True:
        yield i
        i += 1
        if i == 10:
            break



i = gen()
for k in range(10):
    # k = gen()
    print(next(i))