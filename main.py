def sum_of_first_n_numbers(n):
    if n < 1:
        return 0
    return n * (n + 1) // 2

# Example usage:
n = 10
print(f"The sum of the first {n} numbers is: {sum_of_first_n_numbers(n)}")