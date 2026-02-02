import sys
import random

core_prob = 0.80          
core_len_low = 15
core_len_high = 20


def generate_transactions(num_items, num_transactions):
    random.seed(42)

    items = [f"i{i}" for i in range(1, num_items + 1)]

    num_cores = 4
    core_size = min(10, num_items // 2)

    cores = []
    for c in range(num_cores):
        start = (c * (num_items // 4)) % num_items
        core = [items[(start + j) % num_items] for j in range(core_size)]
        cores.append(core)

    transactions = []

    for _ in range(num_transactions):
        t = set()

        #### LLM-assisted code (ChatGPT); all logic and correctness verified by the authors.
        cores_selected = random.sample(cores, random.choice([1, 2]))
        ####

        for core in cores_selected:
            for it in core:
                if random.random() < core_prob:
                    t.add(it)

        for it in items:
            if it not in t and random.random() < 0.25:
                t.add(it)

        for it in items:
            if it not in t and random.random() < 0.05:
                t.add(it)

        target_len = random.randint(core_len_low, core_len_high)

        if len(t) < target_len:
            t.update(random.sample(items, target_len - len(t)))
        elif len(t) > target_len:
            t = set(random.sample(list(t), target_len))

        transactions.append(sorted(t))

    return transactions


def main():
    num_items = int(sys.argv[1])
    num_transactions = int(sys.argv[2])

    transactions = generate_transactions(num_items, num_transactions)

    with open("generated_transactions.dat", "w") as f:
        for t in transactions:
            f.write(" ".join(t) + "\n")


if __name__ == "__main__":
    main()
