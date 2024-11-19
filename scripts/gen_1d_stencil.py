import random

offset_max_n = 20;
offset_distance_max = 20;

# Generate number of offsets
offsets = {}
offsets_n = random.randrange(1, 20);
total_weight = 0;
print(f"offsets_n: {offsets_n}")
for i in range(0, offsets_n):
    print(f" gen offset {i}")
    while True:
        offset = random.randrange(-offset_distance_max, offset_distance_max)
        print(f"Try offset: {offset}")
        if not offset in offsets:
            remaining_weight = 1 - total_weight
            weight = random.uniform(0, remaining_weight)
            if i == offsets_n - 1:
                weight = remaining_weight
            total_weight += weight
            offsets[offset] = weight
            print(f" adding offset {offset}, {weight}")
            break

file = open('examples/gen_1d.stencil', 'w')
file.write(f"Stencil::new(\n[")
for k in offsets:
   file.write(f"[{k}],")
file.write("],\n")
file.write(f"|args: &[f32; {offsets_n}]| {{\n")
for (i, k) in enumerate(offsets):
    file.write(f"args[{i}] * {offsets[k]}f32 + ") 
file.write("0.0\n})")

        


    

