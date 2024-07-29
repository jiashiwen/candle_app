

$ cargo run --release  -- --prompt "Hello there "


cargo run --release  -- --model 2-0.5b  --prompt 'def print_prime(n: int): '
def print_prime(n: int):  # n is the number of primes to be printed
    for i in range(2, n + 1):
        if all(i % j != 0 for j in range(2, i)):
            print(i)


$ cargo run --example qwen --release  -- --model moe-a2.7b --prompt 'def print_prime(n: int): '
def print_prime(n: int):  # n is the number of primes to be printed
    for i in range(2, n + 1):
        if all(i % j != 0 for j in range(2, i)):
            print(i)