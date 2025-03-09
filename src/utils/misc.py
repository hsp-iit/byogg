
def find_monotonous_subsequences(arr):
    subsequences = []
    start = 0

    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            if i - start > 1:
                subsequences.append((start, i - 1))
            start = i

    # Check if the last subsequence extends till the end of the array
    if len(arr) - start > 1:
        subsequences.append((start, len(arr) - 1))

    return subsequences