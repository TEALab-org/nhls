import random

offset_max_n = 40;
offset_distance_max = 20;

# Generate number of offsets
offsets = {}
offsets_n = random.randrange(1, 20);
total_weight = 0;
print(f"offsets_n: {offsets_n}")
for i in range(0, offsets_n):
    print(f" gen offset {i}")
    while True:
        offset_1 = random.randrange(-offset_distance_max, offset_distance_max)
        offset_2 = random.randrange(-offset_distance_max, offset_distance_max)

        print(f"Try offset: {offset_1}, {offset_2}")
        if not (offset_1, offset_2) in offsets:
            remaining_weight = 1 - total_weight
            weight = random.uniform(0, remaining_weight)
            if i == offsets_n - 1:
                weight = remaining_weight
            total_weight += weight
            offsets[(offset_1, offset_2)] = weight
            print(f" adding offset {offset_1}, {offset_2}, {weight}")
            break

file = open('examples/gen_2d.stencil', 'w')
file.write(f"Stencil::new(\n[")
for (o1, o2) in offsets:
   file.write(f"[{o1}, {o2}],")
file.write("],\n")
file.write(f"|args: &[f32; {offsets_n}]| {{\n")
for (i, k) in enumerate(offsets):
    file.write(f"args[{i}] * {offsets[k]}f32 + ") 
file.write("0.0\n})")
